import os
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
import Utils.MinLA as minla
from Baseline.lower_bound import calculate_lower_obj_bound
from Baseline.spectral_sequencing import spectral_sequencing
from Baseline.successive_augmentation import successive_augmentation

DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_DIR = "Results/quantum_experiment"

def read_dataset():
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

def run_successive_augmentation(G, method='dfs'):
    """Run successive augmentation heuristic."""
    t0 = time.time()
    ordering = successive_augmentation(G, method=method)
    elapsed = time.time() - t0
    
    minla_cost = minla.calculate_min_linear_arrangement(G, ordering)
    return minla_cost, elapsed

def run_spectral_sequencing(G):
    """Run spectral sequencing heuristic."""
    t0 = time.time()
    ordering = spectral_sequencing(G)
    elapsed = time.time() - t0
    
    minla_cost = minla.calculate_min_linear_arrangement(G, ordering)
    return minla_cost, elapsed

def run_experiment():
    datasets = read_dataset()
    
    vertices_count = [5, 10, 15, 20, 25]
    
    all_rows = []

    for vertex_count in vertices_count:
        print(f"\nRunning experiment for graph with {vertex_count} vertices...")
        graphs = datasets[vertex_count]['graphs']

        for graph_id, graph in enumerate(graphs):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()

            lower_bound = calculate_lower_obj_bound(G)
            
            print(f'\n  Graph {graph_id}:')
            
            # ============ Spectral Sequencing ============
            print(f'    Running Spectral Sequencing...')
            try:
                minla_cost_ss, time_ss = run_spectral_sequencing(G)
                approx_ratio_ss = minla_cost_ss / lower_bound
                feasible_ss = True
                print(f'      Spectral: Cost={minla_cost_ss:.2f} | Approx Ratio={approx_ratio_ss:.4f} | Time={time_ss:.4f}s')
            except Exception as e:
                print(f'      Spectral: Failed - {str(e)}')
                minla_cost_ss = None
                approx_ratio_ss = None
                feasible_ss = False
                time_ss = 0
            
            # ============ Successive Augmentation (DFS) ============
            print(f'    Running Successive Augmentation (DFS)...')
            try:
                minla_cost_sa_dfs, time_sa_dfs = run_successive_augmentation(G, method='dfs')
                approx_ratio_sa_dfs = minla_cost_sa_dfs / lower_bound
                feasible_sa_dfs = True
                print(f'      Succ Aug DFS: Cost={minla_cost_sa_dfs:.2f} | Approx Ratio={approx_ratio_sa_dfs:.4f} | Time={time_sa_dfs:.4f}s')
            except Exception as e:
                print(f'      Succ Aug DFS: Failed - {str(e)}')
                minla_cost_sa_dfs = None
                approx_ratio_sa_dfs = None
                feasible_sa_dfs = False
                time_sa_dfs = 0
            
            # ============ Successive Augmentation (BFS) ============
            print(f'    Running Successive Augmentation (BFS)...')
            try:
                minla_cost_sa_bfs, time_sa_bfs = run_successive_augmentation(G, method='bfs')
                approx_ratio_sa_bfs = minla_cost_sa_bfs / lower_bound
                feasible_sa_bfs = True
                print(f'      Succ Aug BFS: Cost={minla_cost_sa_bfs:.2f} | Approx Ratio={approx_ratio_sa_bfs:.4f} | Time={time_sa_bfs:.4f}s')
            except Exception as e:
                print(f'      Succ Aug BFS: Failed - {str(e)}')
                minla_cost_sa_bfs = None
                approx_ratio_sa_bfs = None
                feasible_sa_bfs = False
                time_sa_bfs = 0

            # ============ Record Results ============
            heuristics = [
                ('Spectral Sequencing', feasible_ss, minla_cost_ss, approx_ratio_ss, time_ss),
                ('Successive Augmentation (DFS)', feasible_sa_dfs, minla_cost_sa_dfs, approx_ratio_sa_dfs, time_sa_dfs),
                ('Successive Augmentation (BFS)', feasible_sa_bfs, minla_cost_sa_bfs, approx_ratio_sa_bfs, time_sa_bfs),
            ]
            
            for heuristic_name, feasible, minla_cost, approx_ratio, elapsed_time in heuristics:
                row = {
                    'n': n,
                    'm': m,
                    'graph_id': graph_id,
                    'solver': heuristic_name,
                    'feasible': feasible,
                    'avg_minla_cost': minla_cost,
                    'lower_bound': lower_bound,
                    'approx_ratio': approx_ratio,
                    'time_s': round(elapsed_time, 4),
                    'num_seeds': 1
                }
                all_rows.append(row)
    
    # Save results to CSV
    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"heuristics_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_experiment()
