import os
import sys
import pickle
import time
import pandas as pd
import networkx as nx
import openjij as oj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10
NUM_SWEEPS = 1000

SQA_BETA_GRID = [1, 10, 100, 1000]
SQA_GAMMA_GRID = [1, 10, 100, 1000]
TROTTERS = [2]

def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G

def read_extra_graphs():
    with open(DATASET_PATH, "rb") as f:
        extra_data = pickle.load(f)

    graphs = []
    for group in extra_data.values():
        for graph_data in group["graphs"]:
            graphs.append(graph_data)
    return graphs

def print_result(solver_name, config_count, total_configs, n, graph_id, seed,
                  num_sweeps, beta_range, trotters, gamma, feasible, cost, elapsed):
    status = "OK" if feasible else "--"
    cost_str = f"{cost:6.2f}" if cost is not None else "   N/A"
    print(
        f"[{solver_name}] [{config_count}/{total_configs}] N={n} graph={graph_id} seed={seed:<3} | "
        f"sweeps={num_sweeps:<5} beta=({beta_range[0]:.2e},{beta_range[1]:.2e}) gamma={gamma:<5.2g} "
        f"trotters={trotters:<3} | {status} cost={cost_str} | time={elapsed:.2f}s"
    )
    
def run_experiment():
    graphs = read_extra_graphs()[0]
    
    
    configs = [
        (beta, gamma, trotter, seed) for beta in SQA_BETA_GRID for gamma in SQA_GAMMA_GRID for trotter in TROTTERS for seed in SEEDS
    ]
    
    solver = oj.SQASampler()
    total_configs = len(configs)
    
    for graph in graphs:
        G = convert_graph_data_to_nx(graph)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        graph_id = graph["id"]
        lower_bound = graph["lower_bound"]
        bqm = minla.generate_bqm_instance(G)
        
        bqm.normalize()
    
        for config_count, (beta, gamma, trotter, seed) in enumerate(configs, start=1):
            t0 = time.time()
            sampleset = solver.sample(bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, beta=beta, gamma=gamma, trotter=trotter, seed=seed)
            elapsed = time.time() - t0
            
            best_cost = None
            for sample in sampleset.samples():
                ordering, is_feasible = minla.decode_solution(sample, n)
                if is_feasible:
                    cost = minla.calculate_min_linear_arrangement(G, ordering)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost

            feasible = best_cost is not None
            approx_ratio = best_cost / lower_bound if feasible else None
            
            print_result("OpenJijSQA", config_count, total_configs, n, graph_id, seed,
                NUM_SWEEPS, (beta, beta), trotter, gamma, feasible, best_cost, elapsed)
            
if __name__ == "__main__":
    run_experiment()