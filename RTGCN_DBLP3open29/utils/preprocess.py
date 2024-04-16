
import numpy as np
import dill
import pickle as pkl
import networkx as nx
import random

from sklearn.model_selection import train_test_split
from utils.utilities import run_random_walks_n2v

np.random.seed(123)

def load_graphs(dataset_str,time_step):
    """
    Load graph snapshots given the name of dataset
    """
    data_path=("data/{}/{}".format(dataset_str, "DBLP3.npz"))
    data_dblp = np.load(data_path, allow_pickle=True)
    adj_matrix = data_dblp['adjs']  # nadarry
    attribute_matrix = data_dblp['attmats']  # nadarry
    # label_matrix = data_dblp['labels']  # nadarry
    G=[]
    for i in range(time_step):
        G.append(nx.Graph(adj_matrix[i]))
    adjs = [nx.adjacency_matrix(g) for g in G]
    Attribute_matrix=[]
    for j in range(len(adj_matrix)):
        Attribute_matrix.append(attribute_matrix[:,j,:])

    return G, adjs,Attribute_matrix,data_dblp

def get_context_pairs(graphs, adjs,time_step):
    """
    Load/generate context pairs for each snapshot through random walk sampling
    """
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(time_step):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=20))

    return context_pairs_train

def get_evaluation_data(graphs,time_step):
    """
    Load train/val/test examples to evaluate link prediction performance
    """
    eval_idx = time_step - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2,  test_mask_fraction=0.6)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """
    Splits the edges of the graph into training, validation, and testing sets based on the next graph state.

    Args:
        graph (Graph): The current graph.
        next_graph (Graph): The future state of the graph which contains the next time step edges.
        val_mask_fraction (float): Fraction of edges to use for validation.
        test_mask_fraction (float): Fraction of edges to use for testing.

    Returns:
        tuple: Tuples of arrays containing positive and negative edges for training, validation, and testing.
    """
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
            
def negative_sample(edges_pos, nodes_num, next_graph):
    """
    Generates negative samples by randomly picking pairs of nodes that are not connected in the next_graph.

    Args:
        edges_pos (array): Array of positive edges.
        nodes_num (int): Total number of nodes in the graph.
        next_graph (Graph): Graph object of the next state to ensure negatives are not positive in the next step.

    Returns:
        list: List of negative edges.
    """
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

def get_evaluation_classification_data(dataset,num_nodes,num_time_steps):
    """
    Generate indices for node classification task for different training ratios.

    Args:
        dataset (str): Name of the dataset.
        num_nodes (int): Total number of nodes.
        num_time_steps (int): Total number of time steps considered.

    Returns:
        list: List of train, validation, and test indices for each ratio and time step.
    """
    eval_idx=num_time_steps-2
    eval_path='data/{}/eval_nodeclassification_{}.npz'.format(dataset,str(eval_idx))
    
    train_ratios=[0.3,0.5,0.7]
    datas=[]
    for ratio in train_ratios:
        data=[]
        for i in range(num_time_steps):
            idx_val=random.sample(range(num_nodes),int(num_nodes*0.25))
            remaining=np.setdiff1d(np.array(range(num_nodes)),idx_val)
            idx_train=random.sample(list(remaining),int(num_nodes*ratio))
            idx_test=np.setdiff1d(np.array(remaining),idx_train)
            data.append([idx_train,idx_val,list(idx_test)])
        datas.append(data)
    np.savez(eval_path,data=np.array(datas))
    return datas