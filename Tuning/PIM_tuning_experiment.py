import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler
from dwave.samplers import Neal
from typing import Dict, Tuple, List
import openjij as oj

# Add parent directory to path to import Utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

# Set paths relative to parent directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results")
SEED = 42

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
    np.random.seed(SEED)
    datasets = read_dataset()
    
    # beta_range_min = [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1, 5, 10]
    # beta_range_max = [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1, 5, 10]
    
    beta_range_min = [1e-6]
    beta_range_max = [1]
   
    graph_data = datasets[30]['graphs'][0]
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    bqm = minla.generate_bqm_instance(G)
    # bqm.normalize()
    optimal_cost = graph_data.get('optimal_cost', None)
    
    rows = []
    total_configs = len(beta_range_min) * len(beta_range_max)
    config_count = 0
    num_sweeps = 1000
    
    for beta_min in beta_range_min:
        for beta_max in beta_range_max:
            if beta_min > beta_max:
                total_configs -= 1
                continue

            config_count += 1
            t0 = time.time()
             
            solver = PathIntegralAnnealingSampler()
            
            beta_schedule_type = 'custom'
            # beta_range = (beta_min, beta_max)
            # Hp_field = np.linspace(beta_min, beta_max, num=num_sweeps)
            # Hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)
            Hp_field = np.logspace(beta_min, beta_max, num=num_sweeps)
            Hd_field = np.logspace(beta_max, beta_min, num=num_sweeps)
            
            
            sampleset = solver.sample(
                bqm,
                num_reads=10,
                num_sweeps=num_sweeps,
                beta_schedule_type=beta_schedule_type,
                seed=SEED,
                # beta_range=beta_range
                Hp_field=Hp_field,
                Hd_field=Hd_field
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
            
            row = {
                'n': n,
                'm': m,
                'beta_min': beta_min,
                'beta_max': beta_max,
                'beta_schedule_type': beta_schedule_type,
                'energy': energy,
                'feasible': feasible,
                'minla_cost': minla_cost,
                'optimal_cost': optimal_cost,
                'relative_gap': rel_gap,
                'time_s': round(elapsed, 3),
            }
            rows.append(row)
            
            # Print result for each sample
            status = "✓" if feasible else "✗"
            gap_str = f"{rel_gap:.4f}" if rel_gap is not None else "N/A"
            print(f"[{config_count}/{total_configs}] beta=({beta_min:.2e}, {beta_max:.2e}) | {status} "
                  f"E={energy:12.2f} | cost={minla_cost} | optimal_cost={optimal_cost} | {elapsed:.2f}s")
    
    # Save results to CSV
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"PIM_tuning_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    feasible_count = df['feasible'].sum()
    total_count = len(df)
    feasibility_rate = feasible_count / total_count * 100
    
    print(f"Total configurations tested: {total_count}")
    print(f"Feasible solutions: {feasible_count} ({feasibility_rate:.1f}%)")
    
    if feasible_count > 0:
        best_feasible = df[df['feasible']].nsmallest(1, 'minla_cost')
        print(f"\nBest feasible solution:")
        print(f"  Beta range: ({best_feasible['beta_min'].values[0]}, {best_feasible['beta_max'].values[0]})")
        print(f"  MinLA cost: {best_feasible['minla_cost'].values[0]}")
        print(f"  Relative gap: {best_feasible['relative_gap'].values[0]:.4f}")
        print(f"  Time: {best_feasible['time_s'].values[0]}s")
    
    return df


if __name__ == "__main__":
    df = run_experiment()
    
    # df_results = pd.read_csv("Results/tuning_experiment_20260316_232311.csv")
    # top_10_best = df_results.nsmallest(10, 'energy')
    # for _, row in top_10_best.iterrows():
    #     print(f"Beta range: ({row['beta_min']}, {row['beta_max']}) | Energy: {row['energy']} | MinLA cost: {row['minla_cost']}")