import os
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler
import Utils.MinLA as minla
from typing import Dict, Tuple, List

DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_DIR = "Results"

def read_dataset():
    # read all pickle files in DATASET_PATH
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data

    return datasets

def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    return G

def run_experiment():
    datasets = read_dataset()
    
    vertices_count = [5,10,15,20,25]
    
    for vertex_count in vertices_count:
        print(f"Running experiment for graph with {vertex_count} vertices...")
        graphs = datasets[vertex_count]['graphs']
        for graph in graphs:
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            bqm = minla.generate_bqm_instance(G)
            
            