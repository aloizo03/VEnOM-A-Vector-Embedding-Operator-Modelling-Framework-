import pickle
import networkx as nx
from karateclub.graph_embedding import Graph2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import TUDataset
import os 
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
import os 
from tqdm import tqdm
import joblib
import os
import time
import random
from glob import glob
import gzip
import numpy as np
from collections import defaultdict

def load_and_save_Collab():
    print("Mapping nodes to graphs")
    with open("/opt/dlami/nvme/Results/Data/COLLAB/COLLAB/COLLAB_graph_indicator.txt") as f:
        graph_indicator = [int(line.strip()) for line in f]
    node_to_graph = {i + 1: gid for i, gid in enumerate(graph_indicator)}

    print("Adding graph labels")
    # Graph labels (for evaluation, optional)
    with open("/opt/dlami/nvme/Results/Data/COLLAB/COLLAB/COLLAB_graph_labels.txt") as f:
        graph_labels = [int(line.strip()) for line in f]

    print("Creating graphs")
    # Create graphs
    graphs = defaultdict(lambda: nx.Graph())

    print("Adding nodes to graphs")
    # Add nodes to graphs
    for node_id, graph_id in node_to_graph.items():
        graphs[graph_id].add_node(node_id)

    print("Adding edges to graphs")
    # Add edges
    with open("/opt/dlami/nvme/Results/Data/COLLAB/COLLAB/COLLAB_A.txt") as f:
        for line in f:
            u, v = map(int, line.strip().split(','))
            g_u = node_to_graph[u]
            g_v = node_to_graph[v]
            if g_u == g_v:
                graphs[g_u].add_edge(u, v)

    print("Converting to list")
    # Convert to list
    # Ensure each graph has 0-based consecutive node indexing
    graph_list = [nx.convert_node_labels_to_integers(graphs[i + 1]) for i in range(len(graph_labels))]

    print("Saving the list")
    os.makedirs("/opt/dlami/nvme/Results/Data/COLLAB/Collab_graphs/graphs_edgelist", exist_ok=True)
    for i, g in enumerate(graph_list):
        nx.write_edgelist(g, f"/opt/dlami/nvme/Results/Data/COLLAB/Collab_graphs/graphs_edgelist/graph_{i}.edgelist", data=False)

    with open("/opt/dlami/nvme/Results/Data/COLLAB/Collab_graphs/graphs_edgelist/graph_labels.csv", "w") as f:
        f.write("graph_id,label\n")
        for i, label in enumerate(graph_labels):
            f.write(f"{i},{label}\n")

def load_and_save_TUDataset(dataset_name='PROTEINS_full', 
                            graph_indicator_path=None,
                            graph_label_path=None, 
                            graph_file=None,
                            outpath='out/path', use_torch_geo=False):    
    # Load the "PROTEINS" dataset (alternatively: "PROTEINS_full")
    if use_torch_geo:
        print('Load Dataset {dataset_name} from TUDataset')
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")

        os.makedirs(outpath, exist_ok=True)
        graph_labels = []
        for i, data in enumerate(dataset):
            G = to_networkx(data)
            graph_labels.append(int(data.y.item()))
            nx.write_edgelist(G, os.path.join(outpath, f"graph_{i}.edgelist"), data=False)

        with open(os.path.join(outpath, 'graph_labels.csv'), "w") as f:
            f.write("graph_id,label\n")
            for i, label in enumerate(graph_labels):
                f.write(f"{i},{label}\n")
    else:
        print(f'Load Dataset {dataset_name} locally')
        with open(graph_indicator_path) as f:
            graph_indicator = [int(line.strip()) for line in f]
        node_to_graph = {i + 1: gid for i, gid in enumerate(graph_indicator)}

        print("Adding graph labels")
        # Graph labels (for evaluation, optional)
        with open(graph_label_path) as f:
            graph_labels = [int(line.strip()) for line in f]

        print("Adding nodes to graphs")
        print("Creating graphs")
        # Create graphs
        graphs = defaultdict(lambda: nx.Graph())
        # Add nodes to graphs
        for node_id, graph_id in node_to_graph.items():
            graphs[graph_id].add_node(node_id)

        print("Adding edges to graphs")
        # Add edges
        with open(graph_file) as f:
            for line in f:
                u, v = map(int, line.strip().split(','))
                g_u = node_to_graph[u]
                g_v = node_to_graph[v]
                if g_u == g_v:
                    graphs[g_u].add_edge(u, v)

        print("Converting to list")
        # Convert to list
        # Ensure each graph has 0-based consecutive node indexing
        graph_list = [nx.convert_node_labels_to_integers(graphs[i + 1]) for i in range(len(graph_labels))]

        print("Saving the list")
        os.makedirs(outpath, exist_ok=True)
        for i, g in enumerate(graph_list):
            nx.write_edgelist(g, os.path.join(outpath, f'graph_{i}.edgelist'), data=False)

        with open(os.path.join(outpath, 'graph_labels.csv'), "w") as f:
            f.write("graph_id,label\n")
            for i, label in enumerate(graph_labels):
                f.write(f"{i},{label}\n")


# load_and_save_TUDataset(dataset_name='ENZYMES', 
#                         graph_indicator_path='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_graph_indicator.txt',
#                         graph_label_path='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_graph_labels.txt',
#                         graph_file='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_A.txt',
#                         outpath='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_Graphs')

# load_and_save_TUDataset(dataset_name='ENZYMES', 
#                         graph_indicator_path='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_graph_indicator.txt',
#                         graph_label_path='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_graph_labels.txt',
#                         graph_file='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_A.txt',
#                         outpath='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_Graphs')

# load_and_save_TUDataset(dataset_name='PROTEINS_full',
#                         use_torch_geo=True,
#                         outpath='/opt/dlami/nvme/Results/Data/PROTEINS/PROTEINS_edgelist')

load_and_save_TUDataset(dataset_name='ENZYMES',
                        use_torch_geo=True,
                        outpath='/opt/dlami/nvme/Results/Data/ENZYMES/ENZYMES_Graphs')

load_and_save_TUDataset(dataset_name='TWITTER-Real-Graph-Partial',
                        use_torch_geo=True,
                        outpath='/opt/dlami/nvme/Results/Data/TWITTER-Real-Graph-Partial/TWITTER-Graphs')

