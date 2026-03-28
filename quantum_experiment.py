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
    SEEDS = [42, 123, 456, 789, 999]
    
    start_time = time.time()
    
    datasets = read_dataset()
    
    vertices_count = [5,10,15,20,25]
    num_sweeps = 1000

    beta_min = -9
    beta_max = 0
    
    all_rows = []

    for vertex_count in vertices_count:
        print(f"\nRunning experiment for graph with {vertex_count} vertices...")
        vertex_start_time = time.time()
        graphs = datasets[vertex_count]['graphs']

        feasibility_cnt = 0
        approx_ratios = []

        for graph_id, graph in enumerate(graphs):
            graph_start_time = time.time()
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()

            bqm = minla.generate_bqm_instance(G)
            # bqm.normalize()
            optimal_cost = graph.get('optimal_cost', None)

            # Store results for all seeds
            seed_results = []

            for seed in SEEDS:
                np.random.seed(seed)
                
                t0 = time.time()
                
                # solver = oj.SQASampler()
                
                # sampleset = solver.sample(
                #     bqm,
                #     num_reads=10,
                #     num_sweeps=num_sweeps,
                #     sparse=True,
                #     seed=seed
                # )

                solver = PathIntegralAnnealingSampler()

                beta_schedule_type = 'custom'
                # Hp_field = np.linspace(beta_min, beta_max, num=num_sweeps)
                # Hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)
                
                Hp_field = np.power(np.linspace(0, 1, num_sweeps), 1/2)
                Hd_field = np.ones(num_sweeps)

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

                best = sampleset.first
                energy = best.energy
                ordering, feasible = decode_solution(best.sample, n)
                minla_cost = (
                    minla.calculate_min_linear_arrangement(G, ordering)
                    if feasible else None
                )
                
                rel_gap = (
                    (minla_cost - optimal_cost) / optimal_cost
                    if (feasible and optimal_cost) else None
                )

                seed_results.append({
                    'seed': seed,
                    'energy': energy,
                    'feasible': feasible,
                    'minla_cost': minla_cost,
                    'rel_gap': rel_gap,
                    'time_s': elapsed
                })
            
            # Select best result: prefer feasible solutions, then lowest cost
            best_result = None
            for result in seed_results:
                if best_result is None:
                    best_result = result
                elif result['feasible'] and not best_result['feasible']:
                    best_result = result
                elif result['feasible'] and best_result['feasible']:
                    if result['minla_cost'] < best_result['minla_cost']:
                        best_result = result
                elif not result['feasible'] and not best_result['feasible']:
                    if result['energy'] < best_result['energy']:
                        best_result = result

            # Add best result to output
            approx_ratio = None
            if best_result['feasible']:
                feasibility_cnt += 1
                approx_ratio = best_result['minla_cost'] / optimal_cost
                approx_ratios.append(approx_ratio)

            row = {
                'n': n,
                'm': m,
                'graph_id': graph_id,
                'energy': best_result['energy'],
                'feasible': best_result['feasible'],
                'minla_cost': best_result['minla_cost'],
                'optimal_cost': optimal_cost,
                'approx_ratio': approx_ratio,
                'relative_gap': best_result['rel_gap'],
                'time_s': round(best_result['time_s'], 3),
                'best_seed': best_result['seed'],
                'solver': 'PathIntegralAnnealingSampler'
            }
            all_rows.append(row)

            graph_elapsed = time.time() - graph_start_time
            print(f'  Graph {graph_id}: Feasible={best_result["feasible"]} | Energy={best_result["energy"]} | Optimal={optimal_cost} | Best Seed={best_result["seed"]} | Time={graph_elapsed:.2f}s')
        
        feasibility_rate = feasibility_cnt / len(graphs)
        avg_approx_ratio = sum(approx_ratios) / len(approx_ratios) if approx_ratios else None
        vertex_elapsed = time.time() - vertex_start_time
        print(f'  Feasibility rate: {feasibility_rate:.2%}')
        print(f'  Avg approx ratio: {avg_approx_ratio}')
        print(f'  Time for {vertex_count} vertices: {vertex_elapsed:.2f}s')
    
    # Save results to CSV
    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"quantum_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Display total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*70}")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.2f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_experiment()
            
            