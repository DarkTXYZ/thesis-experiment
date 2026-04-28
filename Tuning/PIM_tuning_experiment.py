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
# SEEDS = [123]
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
                 feasible: bool, energy: float, minla_cost, elapsed: float, seed: int):
    status = "✓" if feasible else "✗"
    energy_str = f"{energy:12.2f}" if energy is not None else "         N/A"
    print(f"[{config_count}/{total_configs}] ({seed}) normalized={normalized} | {space_type:9} | annealing={annealing_type} | beta=({beta_min:.2e}, {beta_max:.2e}) | {status} "
          f"E={energy_str} | cost={minla_cost} | {elapsed:.2f}s")


def run_experiment():
    datasets = read_dataset()
    
    # beta_range_min = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
    beta_range_min = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10])
    # beta_range_min = np.array([0.001])
    # beta_range_max = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
    beta_range_max = np.array([1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10])
    # beta_range_max = np.array([1])
    
    space_types = ['linear']
    # space_types = ['power (2)']
    # annealing_types = ['default']
    annealing_types = ['default', 'fixed_Hp', 'fixed_Hd']
    normalized = [False]
    
    N = 25
    num_graphs = 1
    
    graph_data = datasets[N]['graphs'][0]
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    solver = PathIntegralAnnealingSampler()
    # solver = SimulatedAnnealingSampler()
    num_sweeps = 1000
    
    RESULTS_CSV = os.path.join(RESULTS_DIR, f"PIM_tuning_experiment_{N}v_{num_graphs}graphs_{TIMESTAMP}.csv")

    print(f"Precomputing BQMs for N={N}...")
    bqm_unnorm = minla.generate_bqm_instance(G)
    bqm_norm = bqm_unnorm.copy()
    bqm_norm.normalize()
    
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
        for config_count, (norm, space_type, annealing_type, beta_min, beta_max) in enumerate(configs, 1):
            
            bqm = bqm_norm if norm else bqm_unnorm

            # Run each configuration with all seeds
            for seed_idx, seed in enumerate(SEEDS, 1):
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

                # sampleset = solver.sample(
                #     bqm,
                #     num_reads=10,
                #     num_sweeps=num_sweeps,
                #     beta_schedule_type='custom',
                #     beta_schedule=Hp_field,
                #     seed=seed,
                # )

                elapsed = time.time() - t0

                best_feasible_energy = float('inf')
                best_infeasible_energy = float('inf')
                best_ordering = None

                for sample, energy in sampleset.data(fields=['sample', 'energy']):
                    ordering, is_feasible = minla.decode_solution(sample, n)
                    if is_feasible:
                        if best_feasible_energy > energy:
                            best_feasible_energy = energy
                            best_ordering = ordering
                    else:
                        if best_infeasible_energy > energy:
                            best_infeasible_energy = energy

                best_energy = 0
                best_feasible = True
                minla_cost = None
                if best_feasible_energy < float('inf'):
                    best_energy = best_feasible_energy
                    minla_cost = minla.calculate_min_linear_arrangement(G, best_ordering)
                else:
                    best_energy = best_infeasible_energy
                    best_feasible = False

                row = {
                    'n': n,
                    'm': m,
                    'beta_min': beta_min,
                    'beta_max': beta_max,
                    'bqm_is_normalized': norm,
                    'space_type': space_type,
                    'annealing_type': annealing_type,
                    'seed': seed,
                    'energy': best_energy,
                    'feasible': best_feasible,
                    'minla_cost': minla_cost,
                    'time_s': round(elapsed, 3),
                }

                all_results.append(row)
                processed = config_count

                print_result(config_count * num_seeds, total_configs * num_seeds, norm, space_type, annealing_type, beta_min, beta_max, best_feasible, best_energy, minla_cost, elapsed, seed)
                
    except KeyboardInterrupt:
        print(f"\nInterrupted at config {processed}/{total_configs}. Partial results saved.")

    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to {RESULTS_CSV}")
    print(f"{'='*70}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"Total configurations tested: {len(df)}")
    feasible_count = df['feasible'].sum()
    print(f"Feasible solutions: {feasible_count} ({feasible_count/len(df)*100:.1f}%)")
    print(f"Best energy: {df['energy'].min():.2f}")
    print(f"Average energy: {df['energy'].mean():.2f}")
    print(f"Average time: {df['time_s'].mean():.3f}s")
    
    if feasible_count > 0:
        # Compare across configurations using the actual real-world cost metric!
        best = df[df['feasible'] == True].nsmallest(1, 'minla_cost').iloc[0]
        print(f"\nBest feasible solution (Lowest MinLA cost):")
        print(f"  Space type: {best['space_type']}")
        print(f"  Annealing type: {best['annealing_type']}")
        print(f"  Beta range: ({best['beta_min']}, {best['beta_max']})")
        print(f"  Normalized BQM: {best['bqm_is_normalized']}")
        print(f"  Energy (Internal): {best['energy']:.2f}")
        print(f"  MinLA cost: {best['minla_cost']}")
    print(f"{'='*70}")
    
    return df


def analyze_best_config(df):
    """Find which configuration produces the most feasible solutions across seeds."""
    print("\n" + "=" * 70)
    print(" CONFIG ANALYSIS: Most Feasible Solutions")
    print("=" * 70)
    
    # Group by configuration parameters (excluding seed)
    config_cols = ['n', 'm', 'beta_min', 'beta_max', 'bqm_is_normalized', 'space_type', 'annealing_type']
    config_stats = df.groupby(config_cols).agg({
        'feasible': ['sum', 'count', 'mean']
    }).reset_index()
    
    config_stats.columns = config_cols + ['feasible_count', 'total_runs', 'feasibility_rate']
    
    # Sort by feasible count (descending), then by feasibility rate
    config_stats = config_stats.sort_values(['feasible_count', 'feasibility_rate'], ascending=False)
    
    print(f"\nTop 5 Configurations by Feasibility:")
    print(f"{'Rank':<5} {'Beta Range':<25} {'Space Type':<15} {'Feasible/Total':<15} {'Rate':<8}")
    print("-" * 70)
    
    for rank, (idx, row) in enumerate(config_stats.head(5).iterrows(), 1):
        beta_range = f"({row['beta_min']:.2e}, {row['beta_max']:.2e})"
        feasible_str = f"{int(row['feasible_count'])}/{int(row['total_runs'])}"
        rate = f"{row['feasibility_rate']*100:.1f}%"
        print(f"{rank:<5} {beta_range:<25} {row['space_type']:<15} {feasible_str:<15} {rate:<8}")
    
    # Get the best configuration
    best_config = config_stats.iloc[0]
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION:")
    print(f"{'='*70}")
    print(f"Beta range: ({best_config['beta_min']:.2e}, {best_config['beta_max']:.2e})")
    print(f"Space type: {best_config['space_type']}")
    print(f"Annealing type: {best_config['annealing_type']}")
    print(f"BQM normalized: {best_config['bqm_is_normalized']}")
    print(f"Feasible solutions: {int(best_config['feasible_count'])} out of {int(best_config['total_runs'])} seeds")
    print(f"Feasibility rate: {best_config['feasibility_rate']*100:.1f}%")
    print(f"{'='*70}\n")
    
    return config_stats


if __name__ == "__main__":
    df = run_experiment()
    config_stats = analyze_best_config(df)