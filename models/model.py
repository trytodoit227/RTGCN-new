import torch.nn.functional as F
from utils1 import *
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
import sys
sys.path.append('/home/ynos/Desktop/RTGCN')
BN = True


class RTGCN(nn.Module):
    """
    act: activation function for Structural Role-based GRU
    n_node: number of nodes on the network
    output_dim: output embed size of node embedding
    seq_len: number of graphs
    attn_drop: attention/coefficient matrix dropout rate
    residual: if using short cut or not for GRU network
    neg_weight : the negative sampling ratio
    loss_weight: the hyper-parameter to balance the connective proximity and structural role proximity
    role_num : the number of role sets
    cross_role_num: the number of cross_role sets
    """

    def __init__(self,
                 act,
                 n_node,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 time_step,
                 neg_weight,
                 loss_weight,
                 attn_drop,
                 role_num,
                 cross_role_num,
                 residual=False,
                 ):
        super(RTGCN, self).__init__()

        self.act = act
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim=hidden_dim
        self.num_time_steps = time_step
        self.neg_weight=neg_weight
        self.attn_drop = attn_drop
        self.residual = residual
        self.role_num=role_num
        self.cross_role_num=cross_role_num
        self.bceloss=BCEWithLogitsLoss()
        self.var = {}
        self.hypergnn=nn.ModuleList()
        self.hypergnn.append(HyperGNN(input_dim,hidden_dim))
        self.hypergnn.append(HyperGNN(hidden_dim, output_dim))

        self.gcn=GCN(input_dim,hidden_dim,output_dim,self.attn_drop,concat=False)
        self.evolve_weights=nn.ModuleList()
        self.evolve_weights.append(MatGRUCell( n_node,input_dim, hidden_dim,cross_role_num))
        self.evolve_weights.append(MatGRUCell(n_node, hidden_dim, output_dim, cross_role_num))
        
        self.weight_var1 = nn.Parameter(torch.randn(self.input_dim,  self.hidden_dim))
        self.weight_var2 = nn.Parameter(torch.randn(self.hidden_dim,  self.output_dim))
        
        self.loss_weight=loss_weight
        self.emb_weight=nn.Parameter(torch.ones(1))
        self.emb_cross_weight = nn.Parameter(torch.ones(1))


    def forward(self, data, train_hypergraph,cross_role_hyper,cross_role_laplacian):
        """
        Forward pass of the RTGCN, processing input through both graph and hypergraph components, and combining outputs.
        """
        input_att=data[0][0]
        input_edge=data[1][0]
        embeds = []
        input_hypergraph=train_hypergraph[0]

        weight_var1 = self.weight_var1
        weight_var2 = self.weight_var2

        #gcn
        output1 =self.gcn(input_att,input_edge,weight_var1[:,:self.hidden_dim],weight_var2[:,:self.output_dim])

        #hypergraph
        output2=self.hypergnn[0](input_att, input_hypergraph,weight_var1[:,:self.hidden_dim])
        output2 = self.hypergnn[1](output2, input_hypergraph,weight_var2[:, :self.output_dim])
        output2=F.log_softmax(output2, dim=1)

        #Concatenate the outputs from two types of graphs.
        output = output1+self.emb_weight*output2

        # Add to embeddings list
        embeds.append(output)

        for i in range(1, len(train_hypergraph)):
            #Load model inputs
            input_att1=data[0][i]
            input_edge1=data[1][i]
            input_hypergraph1=train_hypergraph[i]#
            input_cross_hypergraph=cross_role_hyper[i-1]
            input_cross_laplacian=cross_role_laplacian[i-1]
            adj_matrix1=data[2][i]

            #Update GNNs model parameters for the current time step
            weight_var1 = self.evolve_weights[0](adj_matrix1,weight_var1,input_cross_hypergraph)
            weight_var2 = self.evolve_weights[1](adj_matrix1,weight_var2, input_cross_hypergraph)

            #gcn
            gnn_output= self.gcn(input_att1, input_edge1,weight_var1[:,:self.hidden_dim],weight_var2[:,:self.output_dim])  

            #hypergraph
            output2_0 = self.hypergnn[0](input_att1, input_hypergraph1, weight_var1[:,:self.hidden_dim])
            hyper_out = self.hypergnn[1](output2_0, input_hypergraph1, weight_var2[:, :self.output_dim])
            hyper_out = F.log_softmax(hyper_out, dim=1)
            
            #cross_role graph
            output3 = self.hypergnn[0](data[0][i-1], input_cross_laplacian, weight_var1[:,:self.hidden_dim])
            output3 = self.hypergnn[1](output3, input_cross_laplacian, weight_var2[:, :self.output_dim])
            output3 = F.log_softmax(output3, dim=1)

            #Concatenate the outputs from three types of graphs as node embeddings.
            output = gnn_output+self.emb_weight*hyper_out+self.emb_cross_weight*output3
            embeds.append(output)

        #Return a list of node embeddings.
        return embeds

    def get_loss(self, feed_dict ,data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian,list_loss_role):
        """
        Compute the loss for the model based on the predictions and true data.
        """
        node_1, node_2, node_2_negative = feed_dict.values()

        # Obtain a list of node embeddings through forward propagation.
        final_emb = self.forward(data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian) # [N, T, F]

        #Calculate cumulative loss across time steps [0, T-1]
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            emb_t = final_emb[t]  # [N, F]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            # Calculate scores for positive and negative node pairs
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten()

            # Binary cross-entropy loss for positive and negative pairs
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))

            #Calculate Connective Proximity loss
            graphloss = pos_loss + self.neg_weight*neg_loss
            self.graph_loss += graphloss

            #Calculate Structural Role Proximity
            role_loss=0
            calculate_loss=list_loss_role[t]
            for l in calculate_loss:
                node_role_emb=emb_t[l]
                a = node_role_emb/torch.norm(node_role_emb,dim=1,keepdim=True)
                similarity = torch.mm(a,a.T)
                I_mat=torch.ones_like(similarity)

                # Frobenius norm for Structural Role Proximity
                role_loss+=torch.norm(similarity-I_mat)**2/2
                del similarity,node_role_emb
            self.graph_loss+=self.loss_weight*role_loss
            
        return self.graph_loss


    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GCNConv(MessagePassing):
    """ Initialize the GCN convolution layer.
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        dropout (float): Dropout rate.
        concat (bool): Whether to concatenate results.
    """
    def __init__(self,in_channels,out_channels,dropout,concat):
        super(GCNConv,self).__init__(aggr='add')

    def forward(self,x,edge_index,W2):
        """ Forward pass of GCN convolution.
        Args:
            x (Tensor): Node feature matrix (N, in_channels).
            edge_index (Tensor): Edge indices.
            W2 (Tensor): Weight matrix for this layer.
        Returns:
            Tensor: Output after convolution.
        """
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        x=torch.matmul(x,W2)
        row,col=edge_index

        #Calculate the degree matrix
        deg=degree(col,x.size(0),dtype=x.dtype)

        #Calculate the negative one-half power of the degree matrix
        deg_inv_sqrt=deg.pow(-0.5)
        norm=deg_inv_sqrt[row]*deg_inv_sqrt[col]
        return self.propagate(edge_index,x=x,norm=norm)
    def message(self,x_j,norm):
        """ Messages passed between nodes.
        Args:
            x_j (Tensor): Feature matrix of neighboring nodes.
            norm (Tensor): Normalized degree matrix.
        """
        return norm.view(-1,1)*x_j


        



class GCN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout,concat):
        """ Initialize the GCN model.
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden layer.
            output_dim (int): Dimension of output features.
            dropout (float): Dropout rate.
            concat (bool): Whether to concatenate results.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim,dropout,concat)
        self.conv2 = GCNConv(hidden_dim, output_dim,dropout,concat)
        self.prej =nn.Linear(input_dim,output_dim,bias=False)
        self.alpha1 = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index,gnn_weight1,gnn_weight2):
        """ Forward pass of the GCN model.
        Args:
            x (Tensor): Input feature matrix.
            edge_index (Tensor): Edge indices.
            gnn_weight1 (Tensor): Weight matrix for first GCN layer.
            gnn_weight2 (Tensor): Weight matrix for second GCN layer.
        Returns:
            Tensor: Output of the model.
        """
        x0 = self.prej(x)
        x = self.conv1(x, edge_index,gnn_weight1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index,gnn_weight2)
        X = x*self.alpha1+x0*(1-self.alpha1)
        return F.log_softmax(X, dim=1)



#HyperGNN
class HyperGNN(nn.Module):
    def __init__(self, input_dim, output_dim, negative_slope=0.2):
        """ Initialize the HyperGNN model.
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            negative_slope (float): Negative slope for leaky ReLU.
        """
        super(HyperGNN, self).__init__()
        self.negative_slope = negative_slope
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, node_initial_emb, hyp_graph,W):
        """ Forward pass of the HyperGNN model.
        Args:
            node_initial_emb (Tensor): Initial node embeddings.
            hyp_graph (Tensor): Hypergraph incidence matrix.
            W (Tensor): Weight matrix for this layer.
        Returns:
            Tensor: Updated node embeddings.
        """
        rs = hyp_graph @ torch.matmul( node_initial_emb,W)
        rs = (1-self.alpha)*self.proj(node_initial_emb)+rs*self.alpha
        return rs





class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.2 of the paper for the formula.
    """

    def __init__(self, n_node,input_dim, output_dim,cross_role_num):
        """Initialize the GRU cell with matrix-specific gates.
        Args:
            n_node (int): Number of nodes in the graph.
            input_dim (int): Dimensionality of input features per node.
            output_dim (int): Dimensionality of output features per node.
            cross_role_num (int): Number of cross-role interactions to consider.
        """
        super().__init__()

        #Update gate
        self.update = MatGRUGate(n_node,input_dim,
                                 output_dim,
                                 torch.nn.Sigmoid(),cross_role_num=cross_role_num)
        #Reset gate.
        self.reset = MatGRUGate(n_node,input_dim,
                                output_dim,
                                torch.nn.Sigmoid(),cross_role_num=cross_role_num)

        # Candidate memory content uses tanh to ensure the state values remain between -1 and 1.
        self.htilda = MatGRUGate(n_node,input_dim,
                                 output_dim,
                                 torch.nn.Tanh(),cross_role_num=cross_role_num)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the GRU parameters
        """
        reset_parameters(self.named_parameters)

    def forward(self, adj,weight_vars,H):
        """Perform a forward pass of the GRU cell.
        Args:
            adj (Tensor): The adjacency matrix.
            weight_vars (Tensor): The weight variables (hidden states).
            H (Tensor): Incident matrix or matrix representing cross-role interactions.
        Returns:
            Tensor: Updated node embeddings.
        """
        update = self.update(adj, weight_vars,H)
        reset = self.reset(adj, weight_vars,H)
        h_cap = reset * weight_vars
        h_cap = self.htilda(adj, h_cap,H)
        new_Q = (1 - update) * weight_vars + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    For datasets with initial node features, if the dimension of the initial
    features does not equal the number of nodes, dimension matching is required.
    """
    def __init__(self, n_node,rows, cols, activation,cross_role_num):
        """Initialize the matrix GRU gate.
        Args:
            n_node (int): Number of nodes.
            rows (int): Number of rows in the matrix (usually corresponds to the number of nodes).
            cols (int): Number of columns in the matrix (dimensionality of the embeddings).
            activation (Activation): Activation function to apply at the gate.
            cross_role_num (int): Number of cross-role types considered in the gate.
        """
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.Tensor(n_node, cols))
        self.W1=nn.Parameter(torch.Tensor(cross_role_num,cols))
        self.U = nn.Parameter(torch.Tensor(cols, cols))
        self.bias =nn.Parameter(torch.Tensor(rows, cols))

        #Dimensional transformation.
        self.transform = False
        if n_node != rows:
            self.P = nn.Parameter(torch.Tensor(rows, n_node))
            self.transform = True
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, adj, hidden,incident_matrix):
        """Forward pass through the gate.
        Args:
            adj (Tensor): Adjacency matrix of the graph.
            hidden (Tensor): Current hidden state.
            incident_matrix (Tensor): Incident matrix representing cross-role interactions.
        Returns:
            Tensor: Output of the gate after applying the activation function.
        """
        temp = adj.matmul(self.W)
        temp1=incident_matrix.matmul(self.W1)
        if self.transform == True:
            out = self.activation(self.P.matmul(temp) +self.P.matmul(temp1)+ hidden.matmul(self.U) + self.bias)
        else:
            out = self.activation(temp + temp1 + hidden.matmul(self.U) + self.bias)

        return out



