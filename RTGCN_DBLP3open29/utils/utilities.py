import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from utils.random_walk import Graph_RandomWalk

import torch


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """
    Perform random walks on the graph using node2vec's sampling strategy.

    Args:
        graph (networkx.Graph): The graph on which random walks are to be performed.
        adj (np.array): The adjacency matrix of the graph.
        num_walks (int): The number of walks to perform for each node.
        walk_len (int): The length of each random walk.

    Returns:
        dict: A dictionary where keys are start nodes and values are lists of context nodes found in random walks.
    """
    # Initialize a NetworkX graph from the given data
    nx_G = nx.Graph()
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])
    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    # Set up the graph for node2vec processing
    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()

    # Generate walks and collect context pairs within a defined window size
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

def fixed_unigram_candidate_sampler(true_clasees, num_true, num_sampled, unique,  distortion, unigrams):
    """
    Samples negative examples using a unigram distribution with optional distortion.

    Args:
        true_classes (np.array): Array of true class indices.
        num_true (int): Number of true classes per example.
        num_sampled (int): Number of negative classes to sample.
        unique (bool): If true, sample without replacement.
        distortion (float): The distortion to apply to the unigram probabilities.
        unigrams (np.array): Unigram probability distribution.

    Returns:
        list: List of sampled negative examples for each input example.
    """
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):
        dist = copy.deepcopy(unigrams)
        candidate = list(range(len(dist)))
        taboo = true_clasees[i].cpu().tolist()
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)
            dist.pop(tabo)
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist/np.sum(dist))
        samples.append(sample)
    return samples

def to_device(batch, device):
    """
    Transfers each tensor in the given batch to the specified device.

    Args:
        batch (dict): A dictionary containing lists of tensors.
        device (torch.device): The device to which the tensors should be moved.

    Returns:
        dict: A new dictionary with the same structure as 'batch', but with all tensors moved to the specified device.
    """
    feed_dict = copy.deepcopy(batch)

    # Extract tensor lists from the dictionary; assumes keys 'node_1', 'node_2', 'node_2_negative'
    node_1, node_2, node_2_negative = feed_dict.values()

    # Move each tensor list to the specified device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]

    # Return the updated dictionary with all tensors on the new device
    return feed_dict


        



