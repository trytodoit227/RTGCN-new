import torch
import random
import os
import numpy as np
import torch.nn as nn
import math
import scipy.sparse as sparse
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score

def reset_parameters(named_parameters):
    """
    Initializes the parameters of a neural network according to their shape.

    Args:
        named_parameters (iterable): An iterable containing parameter name and parameter tensor pairs.
    """
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False





#refer to https://github.com/iMoonLab/THU-HyperG/blob/master/hyperg/hyperg.py
class HyperG:
    def __init__(self, H, X=None, w=None):
        """ Initial the incident matrix, node feature matrix and hyperedge weight vector of hypergraph
        :param H: scipy coo_matrix of shape (n_nodes, n_edges)
        :param X: numpy array of shape (n_nodes, n_features)
        :param w: numpy array of shape (n_edges,)
        """
        assert sparse.issparse(H)
        assert H.ndim == 2

        self._H = H
        self._n_nodes = self._H.shape[0]
        self._n_edges = self._H.shape[1]

        if X is not None:
            assert isinstance(X, np.ndarray) and X.ndim == 2
            self._X = X
        else:
            self._X = None

        if w is not None:
            self.w = w.reshape(-1)
            assert self.w.shape[0] == self._n_edges
        else:
            self.w = np.ones(self._n_edges)

        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def num_edges(self):
        return self._n_edges

    def num_nodes(self):
        return self._n_nodes

    def incident_matrix(self):
        return self._H

    def hyperedge_weights(self):
        return self.w

    def node_features(self):
        return self._X

    def node_degrees(self):
        if self._DV is None:
            H = self._H.tocsr()
            dv = H.dot(self.w.reshape(-1, 1)).reshape(-1)
            self._DV = sparse.diags(dv, shape=(self._n_nodes, self._n_nodes))
        return self._DV

    def edge_degrees(self):
        if self._DE is None:
            H = self._H.tocsr()
            de = H.sum(axis=0).A.reshape(-1)
            self._DE = sparse.diags(de, shape=(self._n_edges, self._n_edges))
        return self._DE

    def inv_edge_degrees(self):
        if self._INVDE is None:
            self.edge_degrees()
            inv_de = np.power(self._DE.data.reshape(-1), -1.)
            self._INVDE = sparse.diags(inv_de, shape=(self._n_edges, self._n_edges))
        return self._INVDE

    def inv_square_node_degrees(self):
        if self._DV2 is None:
            self.node_degrees()
            dv2 = np.power(self._DV.data.reshape(-1)+1e-6, -0.5)
            self._DV2 = sparse.diags(dv2, shape=(self._n_nodes, self._n_nodes))
        return self._DV2

    def theta_matrix(self):
        if self._THETA is None:
            self.inv_square_node_degrees()
            self.inv_edge_degrees()

            W = sparse.diags(self.w)
            self._THETA = self._DV2.dot(self._H).dot(W).dot(self._INVDE).dot(self._H.T).dot(self._DV2)

        return self._THETA

    def laplacian(self):
        if self._L is None:
            self.theta_matrix()
            self._L = sparse.eye(self._n_nodes) - self._THETA
        return self._L

    def update_hyedge_weights(self, w):
        assert isinstance(w, (np.ndarray, list)), \
            "The hyperedge array should be a numpy.ndarray or list"

        self.w = np.array(w).reshape(-1)
        assert w.shape[0] == self._n_edges

        self._DV = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def update_incident_matrix(self, H):
        assert sparse.issparse(H)
        assert H.ndim == 2
        assert H.shape[0] == self._n_nodes
        assert H.shape[1] == self._n_edges

        # TODO: reset hyperedge weights?

        self._H = H
        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None


def gen_attribute_hg(n_nodes,role_dict2,Role_set,X=None):
    """
    :param attr_dict: dict, eg. {'attri_1': [node_idx_1, node_idx_1, ...], 'attri_2':[...]} (zero-based indexing)
    :param n_nodes: int,
    :param X: numpy array, shape = (n_samples, n_features) (optional)
    :return: instance of HyperG
    """

    if X is not None:
        assert n_nodes == X.shape[0]

    n_edges = len(Role_set)
    node_idx = []
    edge_idx = []

    for i, attr in enumerate(role_dict2):
        nodes = sorted(role_dict2[attr])
        node_idx.extend(nodes)
        idx=Role_set.index(attr)
        edge_idx.extend([idx] * len(nodes))

    node_idx = np.asarray(node_idx)
    edge_idx = np.asarray(edge_idx)
    values = np.ones(node_idx.shape[0])

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    return HyperG(H, X=X),H

def cross_role_hypergraphn_nodes(n_nodes,H,role_dict1,role_dict2,Cross_role_Set,w,delta_t,X=None):
    """
      Creates a hypergraph that connects roles across two time steps.

      Args:
          n_nodes (int): Total number of nodes.
          H (matrix): Current hypergraph adjacency matrix.
          role_dict1 (dict): Role dictionary for the previous time step.
          role_dict2 (dict): Role dictionary for the current time step.
          Cross_role_Set (list): Set of roles to be linked across time steps.
          w (float): Weight parameter for the temporal decay.
          delta_t (int): Time difference between two observations.
          X (array, optional): Node features array.

      Returns:
          tuple: A tuple containing the new role hypergraph and the combined hypergraph.
    """
    if X is not None:
        assert n_nodes == X.shape[0]

    n_edges = len(Cross_role_Set)
    node_idx = []
    edge_idx = []

    for i, attr in enumerate(role_dict2):
        if attr in role_dict1.keys():
            nodes=sorted(role_dict1[attr])
            node_idx.extend(nodes)
            idx=Cross_role_Set.index(attr)
            edge_idx.extend([idx] * len(nodes))

    node_idx = np.asarray(node_idx)
    edge_idx = np.asarray(edge_idx)
    values = torch.ones(node_idx.shape[0])*torch.exp(w*torch.tensor(delta_t,dtype=torch.float)).numpy()

    H1 = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    H_cross_role = H + H1
    return HyperG(H1, X=X),H_cross_role

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Converts a scipy sparse matrix to a PyTorch sparse tensor.

    Args:
        sparse_mx (scipy.sparse): Sparse matrix to convert.

    Returns:
        torch.sparse.FloatTensor: Corresponding sparse tensor in PyTorch.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_to_edge(adjmatrix):
    """
    Converts an adjacency matrix to edge indices and values suitable for sparse tensor creation.

    Args:
        adjmatrix (np.array): The adjacency matrix to convert.

    Returns:
        list: List containing edge indices and values.
    """
    adjmatrix=adjmatrix.tolist()
    tmp_coo=sparse.coo_matrix(adjmatrix)
    values=tmp_coo.data
    indices=np.vstack((tmp_coo.row,tmp_coo.col))
    i=torch.LongTensor(indices)
    v=torch.LongTensor(values)
    return [i,v]

def hypergraph_role_set(train_role_graph,time_step):
    """
    Identifies unique roles and cross-time roles for hypergraphs.

    Args:
        train_role_graph (dict): Dictionary of roles per time step.
        time_step (int): Number of time steps considered.

    Returns:
        tuple: Set of unique roles and cross-time roles.
    """
    role_set=[]
    cross_role_set=[]
    for i in range(time_step):
        role_set.extend(list(train_role_graph[i].keys()))
    Role_set=list(set(role_set))
    for j in range(1,time_step):
        cross_role_set.extend(list(train_role_graph[j].keys()))
    Cross_role_Set=list(set(cross_role_set))
    return Role_set,Cross_role_Set

def evaluate_node_classification(emb,labels,datas):
    """
    Evaluates node classification accuracy and AUC across multiple training sets.

    Args:
        emb (array): Node embeddings.
        labels (array): True labels for nodes.
        datas (list): Training/validation/testing split indices.

    Returns:
        tuple: Tuple containing average accuracies, temporary accuracies, average AUCs, and temporary AUCs.
    """
    labels=np.argmax(labels,1)
    average_accs=[]
    Temp_accs=[]
    
    Temp_aucs=[]
    average_aucs=[]
    
    for train_nodes in datas:
        temp_accs=[]
        temp_aucs=[]
        for t,train_node in enumerate(train_nodes):
            train_vec=emb[train_node[0]].detach().cpu().numpy()
            train_y=labels[train_node[0]]

            val_vec=emb[train_node[1]].detach().cpu().numpy()
            val_y=labels[train_node[1]]

            test_vec=emb[train_node[2]].detach().cpu().numpy()
            test_y=labels[train_node[2]]

            clf=LogisticRegression(multi_class='auto',solver='lbfgs',max_iter=4000)
            clf.fit(train_vec,train_y)

            y_pred=clf.predict(test_vec)
            acc=accuracy_score(test_y,y_pred)
            #Calculate the AUC metric
            test_predict = clf.predict_proba(test_vec)
            val_predict = clf.predict_proba(val_vec)

            test_roc_score = roc_auc_score(test_y, test_predict,multi_class='ovr')
            temp_aucs.append(test_roc_score)
            
            temp_accs.append(acc)
        average_acc=statistics.mean(temp_accs)
        average_accs.append(average_acc)
        Temp_accs.append(temp_accs)
        
        average_auc=statistics.mean(temp_aucs)
        average_aucs.append(average_auc)
        Temp_aucs.append(temp_aucs)
    return average_accs,Temp_accs,average_aucs,Temp_aucs
