import os
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler
import Utils.MinLA as minla
from typing import Dict, Tuple, List
import openjij as oj
from Baseline.lower_bound import calculate_lower_obj_bound

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

def decode_solution(raw_sample: Dict, n: int) -> Tuple[np.ndarray, bool]:
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
    return labels == set(range(1, n + 1))

def run_experiment():
    datasets = read_dataset()
    
    vertices_count = [5, 10, 15, 20, 25]
    num_sweeps = 1000
    num_seeds = 5

    beta_min = 1e-9
    beta_max = 1
    
    all_rows = []

    for vertex_count in vertices_count:
        print(f"\nRunning experiment for graph with {vertex_count} vertices...")
        graphs = datasets[vertex_count]['graphs']

        feasibility_cnt = 0
        approx_ratios = []
        total_time = 0

        for graph_id, graph in enumerate(graphs):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()

            bqm = minla.generate_bqm_instance(G)
            lower_bound = calculate_lower_obj_bound(G)

            # Collect best feasible solutions from each seed
            best_feasible_costs = []
            total_elapsed = 0

            for seed in range(num_seeds):
                t0 = time.time()
                
                solver = PathIntegralAnnealingSampler()

                beta_schedule_type = 'custom'
                Hp_field = np.linspace(0, 1, num=num_sweeps)
                Hd_field = np.linspace(1, 0, num=num_sweeps)
                
                sampleset = solver.sample(
                    bqm,
                    num_reads=10,
                    num_sweeps=num_sweeps,
                    beta_schedule_type=beta_schedule_type,
                    Hp_field=Hp_field,
                    Hd_field=Hd_field,
                    seed=seed
                )
                
                elapsed = time.time() - t0
                total_elapsed += elapsed

                # Find the best feasible solution from this seed
                best_feasible_cost = None
                best_energy = None
                for sample in sampleset.samples():
                    ordering, feasible = decode_solution(sample, n)
                    if feasible:
                        minla_cost = minla.calculate_min_linear_arrangement(G, ordering)
                        if best_feasible_cost is None or minla_cost < best_feasible_cost:
                            best_feasible_cost = minla_cost
                            best_energy = sampleset.data_vectors['energy'][list(sampleset.samples()).index(sample)]
                
                if best_feasible_cost is not None:
                    best_feasible_costs.append(best_feasible_cost)

            total_time = total_elapsed

            # Calculate average of best feasible solutions
            if best_feasible_costs:
                avg_minla_cost = np.mean(best_feasible_costs)
                feasibility_cnt += 1
                approx_ratio = avg_minla_cost / lower_bound
                approx_ratios.append(approx_ratio)
                feasible = True
            else:
                avg_minla_cost = None
                approx_ratio = None
                feasible = False

            row = {
                'n': n,
                'm': m,
                'graph_id': graph_id,
                'feasible': feasible,
                'avg_minla_cost': avg_minla_cost,
                'lower_bound': lower_bound,
                'approx_ratio': approx_ratio,
                'time_s': round(total_time, 3),
                'num_seeds': num_seeds,
                'solver': solver.__class__.__name__
            }
            all_rows.append(row)

            if feasible:
                print(f'  Graph {graph_id}: Feasible={feasible} | Avg Cost={avg_minla_cost:.2f} | Lower Bound={lower_bound} | Approx Ratio={approx_ratio:.4f} | Time={total_time:.2f}s')
            else:
                print(f'  Graph {graph_id}: Feasible={feasible} | Lower Bound={lower_bound} | Time={total_time:.2f}s')
        
        feasibility_rate = feasibility_cnt / len(graphs)
        avg_approx_ratio = sum(approx_ratios) / len(approx_ratios) if approx_ratios else None
        print(f'  Feasibility rate: {feasibility_rate:.2%}')
        print(f'  Avg approx ratio: {avg_approx_ratio}')
    
    # Save results to CSV
    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"quantum_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_experiment()
            
            