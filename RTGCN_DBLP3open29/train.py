import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import argparse
import networkx as nx
import pandas as pd
import scipy
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data,get_evaluation_classification_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import RTGCN
from collections import defaultdict
from utils1 import *



import torch
torch.autograd.set_detect_anomaly(True)

def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1) 
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG

 # Experimental settings
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=3,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--dataset', type=str, nargs='?', default='DBLP3',
                        help='dataset name')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--batch_size', type=int, nargs='?', default=256,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=False,
                    help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=50,
                        help="patient")

    # Additional model and training hyperparameters
    parser.add_argument('--node_num', type=int, default=4257,
                        help='Number of node')
    parser.add_argument('--input_dim', type=int, default=100,
                        help='Number of input dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Number of hidden dimension')
    parser.add_argument('--output_dim', type=int, default=24,
                        help='Number of output dimension')
    parser.add_argument('--task', type=str, default='link prediction',
                        help='Name of task')

    # Additional tunable hyperparameters
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, nargs='?', default=True,help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10, help='# negative samples per positive')

    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,help='Walk length for random walk sampling')

    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=10,help='Weightage for negative samples')
    parser.add_argument('--loss_weight', type=float, nargs='?', default=1,help='Weightage for role loss')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.008,help='Initial learning rate for self-attention model.')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0003,help='Initial learning rate for self-attention model.')
    parser.add_argument('--window', type=int, nargs='?', default=-1,help='Window for temporal attention (default : -1 => full)')
    args = parser.parse_args()
    print(args)
    task=args.task

    # Setup device for model training
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    #load  structural role data
    role_path = './data/DBLP3/DBLP3_wl_nc.pkl'
    train_role_graph = pd.read_pickle(role_path)

    # Process structural role data into a list
    list_loss_role=[]
    for h in train_role_graph:
        list_g=[]
        for g in train_role_graph[h]:
            list_g.append(torch.tensor(list(map(int,train_role_graph[h][g]))).to(device))
        list_loss_role.append(list_g)

    #load graph data
    graphs, adjs, feats, data_dblp = load_graphs(args.dataset,args.time_steps)

    # Construct model inputs
    adj_matrix = data_dblp['adjs']
    attribute_matrix = data_dblp['attmats']
    labels=data_dblp['labels']
    Data_dblp=[]
    Data_dblp.append([torch.from_numpy(attribute_matrix[:,i,:]).float().to(device) for i in range(args.time_steps)])
    Data_dblp.append([adj_to_edge(adj_matrix[j,:,:])[0].to(device) for j in range(args.time_steps)])
    Data_dblp.append([scipy_sparse_mat_to_torch_sparse_tensor(adjs[w]).to(device) for w in range(args.time_steps)])
    
    # Contruct role hypergraph
    train_hypergraph = []
    cross_role_hyper = []
    cross_role_laplacian=[]# leanth=time_step-1
    Role_set, Cross_role_Set = hypergraph_role_set(train_role_graph, args.time_steps)
    for i in range(args.time_steps):
        Role_hyper, H = gen_attribute_hg(args.node_num, train_role_graph[i], Role_set, X=None)
        train_hypergraph += [scipy_sparse_mat_to_torch_sparse_tensor(Role_hyper.laplacian()).to(device)]
        if i > 0:
            previous_role_hypergraph,Cross_role_hypergraph=cross_role_hypergraphn_nodes(args.node_num, H, train_role_graph[i - 1], train_role_graph[i], Role_set, w=-11,delta_t=1, X=None)
            cross_role_hyper += [scipy_sparse_mat_to_torch_sparse_tensor(Cross_role_hypergraph).to(device)]
            cross_role_laplacian += [scipy_sparse_mat_to_torch_sparse_tensor(previous_role_hypergraph.laplacian()).to(device)]

    del adj_matrix,attribute_matrix

    # One-hot encoding for initial node features
    if args.featureless == True:
        feats = [scipy.sparse.identity(adjs[args.time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[args.time_steps - 1].shape[0]]

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    context_pairs_train = get_context_pairs(graphs, adjs,args.time_steps)

    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
        test_edges_pos, test_edges_neg = get_evaluation_data(graphs,args.time_steps)
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))
    
    if task=='node classification':
        #load evluation data for node classification
        train_nodes=get_evaluation_classification_data(args.dataset,args.node_num,args.time_steps)
    
    
    # Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
    # inductive testing.
    new_G = inductive_graph(graphs[args.time_steps-2], graphs[args.time_steps-1])
    graphs[args.time_steps-1] = new_G
    adjs[args.time_steps-1] = nx.adjacency_matrix(new_G)

    # build dataloader and model
    dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=0,
                            collate_fn=MyDataset.collate_fn)
    model = RTGCN(act=nn.ELU(),
                          n_node=args.node_num,
                          input_dim=args.input_dim,
                          output_dim=args.output_dim,
                          hidden_dim=args.hidden_dim,
                          time_step=args.time_steps,
                          neg_weight=args.neg_weight,
                          loss_weight=args.loss_weight,
                          attn_drop=0.0,
                          residual=False,
                          role_num=len(Role_set),
                          cross_role_num=len(Role_set))
    model.cuda(device=device)
    # model.to (device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    epochs_test_result = defaultdict(lambda: [])
    epochs_val_result = defaultdict(lambda: [])
    best_epoch_val = 0
    epochs_embeds = []
    patient = 0
    for epoch in range(args.epochs):
        model.train()
        it=0
        epochloss=0.0
        epoch_loss = []
        for idx,feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = model.get_loss(feed_dict, Data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian,list_loss_role)
            loss.backward()
            opt.step()
            epochloss +=loss.item()
            it += 1
        epochloss /= it
        epoch_loss.append(epochloss)
        dataset.test_reset()

        #For validation
        model.eval()
        all_emb = model.forward(Data_dblp, train_hypergraph, cross_role_hyper,cross_role_laplacian)

        # Detach the node embedding at t-1 snapshots used for validation
        emb=all_emb[-2].detach().cpu().numpy()
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                            train_edges_neg,
                                                            val_edges_pos, 
                                                            val_edges_neg, 
                                                            test_edges_pos,
                                                            test_edges_neg, 
                                                            emb, 
                                                            emb)
        epoch_auc_val = val_results["HAD"][0]
        epoch_auc_test = test_results["HAD"][0]
        epochs_test_result["HAD"].append(epoch_auc_test)
        epochs_val_result["HAD"].append(epoch_auc_val)

        #Preserve the optimal model parameters and node embeddings.
        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            epochs_embeds=emb
            epochs_all_embeds=all_emb
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.4f} Test AUC {:.4f}".format(epoch, 
                                                                np.mean(epoch_loss),
                                                                epoch_auc_val, 
                                                                epoch_auc_test))
    # Test Best Model

    # Find the epoch which had the highest validation AUC score
    best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"]))
    print("Best epoch ", best_epoch)

    val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                        train_edges_neg,
                                                        val_edges_pos, 
                                                        val_edges_neg, 
                                                        test_edges_pos,
                                                        test_edges_neg, 
                                                        epochs_embeds,
                                                        epochs_embeds)
    auc_val = val_results["HAD"][0]
    auc_test = test_results["HAD"][0]
    print("Best Test AUC = {:.4f},F1={:.4f},AUC {:.4f}".format(auc_test,test_results["HAD"][1],test_results["HAD"][2]))

    # Node classification task
    if task=='node classification':
        average_accs,temp_accs,average_aucs,temp_aucs=evaluate_node_classification(epochs_all_embeds[-1],labels,train_nodes)
        print('node classification acc [train_ratios=0.3,0.7]:',[average_accs[0],average_accs[2]])
        print(temp_accs)
        print('node classification auc [train_ratios=0.3,0.7]:',[average_aucs[0],average_aucs[2]])
        print(temp_aucs)


                







