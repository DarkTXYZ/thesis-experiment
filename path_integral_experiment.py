from typing import Dict, Tuple
import dimod
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

    bqm = minla.generate_bqm_instance(nx.path_graph(10))
    config = {
        "num_reads": 100,
        "num_sweeps": 1000,
        "seed": 42,
        "beta_schedule_type": "geometric",
    }

    def run(cfg):
        sampler = PathIntegralAnnealingSampler()
        ss = sampler.sample(bqm, **cfg)
        return ss.info, ss

    # Test chain_coupler_strength with NEGATIVE values
    # In C++, pNotJoin = exp(2 * invTempJchain * nOverlaps * invTemp / numTrotterSlices)
    # For domain merging to occur, pNotJoin must be < 1, requiring negative chain_coupler_strength.
    # The hardware constructor uses Jchain = -1.8 (ferromagnetic).
    # In the simple case, numTrotterSlices cancels: pNotJoin = exp(2 * chain_coupler_strength * invTemp)

    num_vars = len(bqm.variables)
    qpc = 4
    assert num_vars % qpc == 0, f"num_vars={num_vars} not divisible by qpc={qpc}"

    base = dict(
        seed=123, num_reads=50, num_sweeps_per_beta=1,
        beta_schedule_type="custom",
        Hp_field=np.geomspace(0.1, 4.0, 100),
        Hd_field=np.linspace(5.0, 0.0, 100),
        qubits_per_chain=qpc,
        qubits_per_update=qpc,
    )

    print("=" * 60)
    print("TEST 1: Positive chain_coupler_strength (should have NO effect)")
    print("=" * 60)
    cfgA = dict(**base, chain_coupler_strength=0.01)
    cfgB = dict(**base, chain_coupler_strength=100.0)
    _, ssA = run(cfgA)
    _, ssB = run(cfgB)
    same1 = np.array_equal(ssA.record.sample, ssB.record.sample)
    print(f"  ccs=0.01  → mean={ssA.record.energy.mean():.2f}, best={ssA.first.energy}")
    print(f"  ccs=100   → mean={ssB.record.energy.mean():.2f}, best={ssB.first.energy}")
    print(f"  Samples identical: {same1}")

    print()
    print("=" * 60)
    print("TEST 2: Negative chain_coupler_strength (SHOULD have effect)")
    print("=" * 60)
    cfgC = dict(**base, chain_coupler_strength=-0.1)
    cfgD = dict(**base, chain_coupler_strength=-5.0)
    _, ssC = run(cfgC)
    _, ssD = run(cfgD)
    same2 = np.array_equal(ssC.record.sample, ssD.record.sample)
    print(f"  ccs=-0.1  → mean={ssC.record.energy.mean():.2f}, best={ssC.first.energy}")
    print(f"  ccs=-5.0  → mean={ssD.record.energy.mean():.2f}, best={ssD.first.energy}")
    print(f"  Samples identical: {same2}")

    print()
    print("=" * 60)
    print("TEST 3: Positive vs Negative chain_coupler_strength")
    print("=" * 60)
    cfgE = dict(**base, chain_coupler_strength=1.0)
    cfgF = dict(**base, chain_coupler_strength=-1.8)  # D-Wave hardware default
    _, ssE = run(cfgE)
    _, ssF = run(cfgF)
    same3 = np.array_equal(ssE.record.sample, ssF.record.sample)
    print(f"  ccs=+1.0  → mean={ssE.record.energy.mean():.2f}, best={ssE.first.energy}")
    print(f"  ccs=-1.8  → mean={ssF.record.energy.mean():.2f}, best={ssF.first.energy}")
    print(f"  Samples identical: {same3}")

    # datasets = read_synthetic_graphs()
    
    # for size, dataset in datasets.items():
    #     experiment_results = []
        
    #     graph0_data = dataset['graphs'][0]
    #     edgelist = graph0_data['edges']
    #     graph0 = nx.from_edgelist(edgelist)
        
    #     bqm = minla.generate_bqm_instance(graph0)
        
    #     print('Number of variables: ' + str(len(bqm.variables)))
    #     print('Number of interactions: ' + str(bqm.num_interactions))
        
    #     sampleset = solve(graph0)
        
    #     best_sample = sampleset.first.sample
    #     energy = sampleset.first.energy
    #     ordering, is_feasible = decode_solution(best_sample, graph0.number_of_nodes())
        
    #     print(f"Graph size: {size}, Feasible: {is_feasible}, Energy: {energy}")
        
    #     break
    #     # experiment(size, dataset['graphs'])
        
    