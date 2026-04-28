import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
from typing import Dict, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results/tuning_experiment")
SEEDS = [42, 123, 456, 789, 999]
# SEEDS = [42]
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')


def read_dataset():
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data
    return datasets

def generate_field_beta(
    space_type: str, 
    annealing_type: str, 
    num_sweeps: int, 
    start: float = 0.0, 
    end: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an annealing schedule across a defined [start, end] range."""
    Hp_field = None
    Hd_field = None
    
    if space_type == 'linear':
        Hp_field = np.linspace(start, end, num_sweeps)
    elif space_type == 'geometric':
        Hp_field = np.geomspace(max(start, 1e-9), end, num_sweeps)
    elif space_type == 'power (1/2)':
        Hp_field = np.linspace(start, end, num_sweeps) ** 0.5
    elif space_type == 'power (2)':
        Hp_field = np.linspace(start, end, num_sweeps) ** 2
        
    Hd_field = (start + end) - Hp_field
    
    if annealing_type == 'fixed_Hp':
        Hp_field = np.full(num_sweeps, end)
    elif annealing_type == 'fixed_Hd':
        Hd_field = np.full(num_sweeps, end)

    return Hp_field, Hd_field


def print_result(config_count: int, total_configs: int, normalized: bool, space_type: str, annealing_type: str, beta_min: float, beta_max: float,
                 feasible: bool, energy: float, minla_cost, optimal_cost, elapsed: float, seed: int):
    status = "✓" if feasible else "✗"
    energy_str = f"{energy:12.2f}" if energy is not None else "         N/A"
    print(f"[{config_count}/{total_configs}] ({seed}) normalized={normalized} | {space_type:9} | annealing={annealing_type} | beta=({beta_min:.2e}, {beta_max:.2e}) | {status} "
          f"E={energy_str} | cost={minla_cost} | optimal_cost={optimal_cost} | {elapsed:.2f}s")


def run_experiment():
    datasets = read_dataset()
    
    # beta_range_min = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
    beta_range_min = np.array([1e-9])
    # beta_range_max = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])

    beta_range_max = np.array([1])
    
    space_types = ['linear', 'power (1/2)', 'power (2)', 'geometric']
    # space_types = ['linear']
    # annealing_types = ['default', 'fixed_Hp', 'fixed_Hd']
    annealing_types = ['default']
    normalized = [False]
    
    N = 25
    num_graphs = 10
    
    solver = PathIntegralAnnealingSampler()
    # solver = SimulatedAnnealingSampler()
    num_sweeps = 1000
    
    RESULTS_CSV = os.path.join(RESULTS_DIR, f"PIM_tuning_experiment_{N}v_{num_graphs}graphs_{TIMESTAMP}.csv")
    
    configs = []
    for norm in normalized:
        for annealing_type in annealing_types:
            for space_type in space_types:
                for beta_min in beta_range_min:
                    for beta_max in beta_range_max:
                        if beta_min > beta_max:
                            continue
                        configs.append((norm, space_type, annealing_type, beta_min, beta_max))
    
    total_configs = len(configs)
    num_seeds = len(SEEDS)
    all_results = []
    processed = 0
    
    try:
        # Iterate through all 10 graphs
        for graph_idx in range(num_graphs):
            graph_data = datasets[N]['graphs'][graph_idx]
            G = nx.Graph()
            G.add_nodes_from(range(graph_data['num_vertices']))
            G.add_edges_from(graph_data['edges'])
            n = G.number_of_nodes()
            m = G.number_of_edges()
            optimal_cost = graph_data.get('optimal_cost', None)
            
            for config_count, (norm, space_type, annealing_type, beta_min, beta_max) in enumerate(configs, 1):
                # Run each configuration with all 5 seeds
                for seed_idx, seed in enumerate(SEEDS, 1):
                    bqm = minla.generate_bqm_instance(G)
                    if norm:
                        bqm.normalize()

                    t0 = time.time()

                    Hp_field, Hd_field = generate_field_beta(space_type, annealing_type, num_sweeps, beta_min, beta_max)

                    sampleset = solver.sample(
                        bqm,
                        num_reads=10,
                        num_sweeps=num_sweeps,
                        beta_schedule_type='custom',
                        seed=seed,
                        Hp_field=Hp_field,
                        Hd_field=Hd_field
                    )

                    elapsed = time.time() - t0
                    
                    best_feasible_cost = None
                    best_energy = None
                    feasibility_count = 0
                    for sample in sampleset.samples():
                        ordering, feasible = minla.decode_solution(sample, n)
                        if feasible:
                            minla_cost = minla.calculate_min_linear_arrangement(G, ordering)
                            feasibility_count += 1
                            if best_feasible_cost is None or minla_cost < best_feasible_cost:
                                best_feasible_cost = minla_cost
                                best_energy = sampleset.data_vectors['energy'][list(sampleset.samples()).index(sample)]
                    
                    row = {
                        'graph_idx': graph_idx,
                        'n': n,
                        'm': m,
                        'beta_min': beta_min,
                        'beta_max': beta_max,
                        'bqm_is_normalized': norm,
                        'space_type': space_type,
                        'annealing_type': annealing_type,
                        'seed': seed,
                        'energy': best_energy,
                        'feasible': best_feasible_cost is not None,
                        'minla_cost': best_feasible_cost,
                        'feasibility_count': feasibility_count,
                        'time_s': round(elapsed, 3),
                    }

                    all_results.append(row)
                    processed = config_count

                    print_result(config_count * num_seeds, total_configs * num_seeds, norm, space_type, annealing_type, 
                                 beta_min, beta_max, best_feasible_cost is not None, best_energy, 
                                 best_feasible_cost, optimal_cost, elapsed, seed)
                
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results saved.")
    
    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to {RESULTS_CSV}")
    print(f"{'='*70}")
    
    # Aggregate results by schedule type across all graphs and seeds
    print("\n" + "=" * 70)
    print(" AGGREGATING RESULTS ACROSS GRAPHS AND SEEDS")
    print("=" * 70)
    
    aggregated = []
    for (graph_idx, space_type), group in df.groupby(['graph_idx', 'space_type']):
        feasible_runs = group[group['feasible'] == True]
        
        agg_row = {
            'graph_idx': graph_idx,
            'space_type': space_type,
            'feasibility_rate': len(feasible_runs) / len(group),  # Across 5 seeds
            'mean_minla_cost': feasible_runs['minla_cost'].mean() if len(feasible_runs) > 0 else float('inf'),
            'mean_runtime': group['time_s'].mean(),
            'num_feasible': len(feasible_runs),
            'num_runs': len(group)
        }
        aggregated.append(agg_row)
    
    agg_df = pd.DataFrame(aggregated)
    
    # Aggregate across all graphs for each schedule type (final decision)
    final_summary = []
    for space_type in space_types:
        space_results = agg_df[agg_df['space_type'] == space_type]
        
        final_row = {
            'space_type': space_type,
            'mean_feasibility_rate': space_results['feasibility_rate'].mean(),
            'mean_minla_cost': space_results['mean_minla_cost'].mean(),
            'mean_runtime': space_results['mean_runtime'].mean(),
        }
        final_summary.append(final_row)
    
    final_df = pd.DataFrame(final_summary)
    
    # Select best schedule using priority rule:
    # 1. Highest feasibility rate
    # 2. Then lowest mean feasible MinLA cost
    # 3. Then lower runtime
    final_df = final_df.sort_values(
        by=['mean_feasibility_rate', 'mean_minla_cost', 'mean_runtime'],
        ascending=[False, True, True]
    )
    
    best_schedule = final_df.iloc[0]
    
    print("\n" + "=" * 70)
    print(" FINAL SCHEDULE SELECTION")
    print("=" * 70)
    print(final_df.to_string(index=False))
    print("\n✓ Selected schedule: {} (feasibility: {:.1%}, mean_minla_cost: {:.2f}, runtime: {:.3f}s)".format(
        best_schedule['space_type'],
        best_schedule['mean_feasibility_rate'],
        best_schedule['mean_minla_cost'],
        best_schedule['mean_runtime']
    ))
    print("=" * 70)
    
    return df

if __name__ == "__main__":
    df = run_experiment()