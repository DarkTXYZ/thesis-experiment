from typing import Dict, Tuple
from dwave.samplers import PathIntegralAnnealingSampler
from dwave.embedding.chain_strength import uniform_torque_compensation
import numpy as np
import Utils.MinLA as minla
import networkx as nx

import os
import pickle
import json

with open("config.json", "r") as jsonfile:
    CONFIG = json.load(jsonfile)

def read_synthetic_graphs():
    synthetic_path = CONFIG['synthetic_dataset_path']
    
    datasets = {}
    for filename in os.listdir(synthetic_path):
        if filename.endswith(".pkl"):
            # extract name and size from filename
            name_size = filename[:-4].split('_')
            size = int(name_size[1][1:])
            
            if size in CONFIG['vertex_counts']:
                datasets[size] = []
            
            with open(os.path.join(synthetic_path, filename), 'rb') as f:
                graph = pickle.load(f)
                datasets[size] = graph
                
    return datasets

def solve(graph: nx.Graph):
    solver = PathIntegralAnnealingSampler()
    
    bqm = minla.generate_bqm_instance(graph)
    chain_strength = uniform_torque_compensation(bqm)
    
    sampleset = solver.sample(
        bqm = bqm,
        num_reads = CONFIG['num_reads'],
        num_sweeps = CONFIG['num_sweeps'],
        seed = CONFIG['seed'],
        beta_schedule_type = CONFIG['beta_schedule_type'],
        chain_coupler_strength=chain_strength
    )
    
    return sampleset

def decode_solution(raw_sample: Dict, num_nodes: int) -> Tuple[np.ndarray, bool]:
    n = num_nodes
    sol = np.array([
        [raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
        for u in range(n)
    ], dtype=int)
    
    is_feasible = check_feasibility(sol, n)
    ordering = np.sum(sol, axis=1)
    
    return ordering, is_feasible

def check_feasibility(sol: np.ndarray, n: int) -> bool:
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return False
    
    labels = set(np.sum(sol, axis=1))
    if labels != set(range(1, n + 1)):
        return False
    
    return True
    

def experiment(size, dataset):
    id = 0
    for graph_data in dataset:
        edgelist = graph_data['edges']
        graph = nx.from_edgelist(edgelist)
        sampleset = solve(graph)
        
        _, is_feasible = decode_solution(sampleset.first.sample, graph.number_of_nodes())
        print(f"Graph id: {id}, Feasible: {is_feasible}, Energy: {sampleset.first.energy}")
        id += 1
        
    # return experiment results for this size: N, M, feasibility rate, success rate (feasible and within 5% of baseline), average relative gaps, std related gaps.
    
if __name__ == "__main__":
    datasets = read_synthetic_graphs()
    
    for size, dataset in datasets.items():
        experiment_results = []
        
        graph0_data = dataset['graphs'][0]
        edgelist = graph0_data['edges']
        graph0 = nx.from_edgelist(edgelist)
        
        bqm = minla.generate_bqm_instance(graph0)
        
        print('Number of variables: ' + str(len(bqm.variables)))
        print('Number of interactions: ' + str(bqm.num_interactions))
        
        sampleset = solve(graph0)
        
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy
        ordering, is_feasible = decode_solution(best_sample, graph0.number_of_nodes())
        
        print(f"Graph size: {size}, Feasible: {is_feasible}, Energy: {energy}")
        
        break
        # experiment(size, dataset['graphs'])
        
    