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
# SEEDS = [42, 123, 456, 789, 999]
SEEDS = [42]
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
    
    beta_range_min = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
    # beta_range_min = np.array([1e-9])
    beta_range_max = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])

    # beta_range_max = np.array([1])
    
    # space_types = ['linear', 'power (1/2)', 'power (2)', 'geometric']
    space_types = ['linear']
    # annealing_types = ['default', 'fixed_Hp', 'fixed_Hd']
    annealing_types = ['default']
    normalized = [False]
    
    N = 25
    num_graphs = 1
    
    graph_data = datasets[N]['graphs'][0]
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # solver = PathIntegralAnnealingSampler()
    solver = SimulatedAnnealingSampler()
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
        for config_count, (norm, space_type, annealing_type, beta_min, beta_max) in enumerate(configs, 1):
            # Run each configuration with all 5 seeds
            for seed_idx, seed in enumerate(SEEDS, 1):
                bqm = minla.generate_bqm_instance(G)
                if norm:
                    bqm.normalize()
                    
                optimal_cost = graph_data.get('optimal_cost', None)

                t0 = time.time()

                Hp_field, Hd_field = generate_field_beta(space_type, annealing_type, num_sweeps, beta_min, beta_max)

                # sampleset = solver.sample(
                #     bqm,
                #     num_reads=10,
                #     num_sweeps=num_sweeps,
                #     beta_schedule_type='custom',
                #     seed=seed,
                #     Hp_field=Hp_field,
                #     Hd_field=Hd_field
                # )

                sampleset = solver.sample(
                    bqm,
                    num_reads=10,
                    num_sweeps=num_sweeps,
                    beta_schedule_type='custom',
                    beta_schedule=Hp_field,
                    seed=seed,
                )

                elapsed = time.time() - t0

                best_feasible_sample = None
                best_feasible_energy = float('inf')
                best_feasible_ordering = None
                
                best_infeasible_sample = None
                best_infeasible_energy = float('inf')

                for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']):
                    ordering, feasible = minla.decode_solution(sample, n)
                    if feasible:
                        if energy < best_feasible_energy:
                            best_feasible_sample = sample
                            best_feasible_energy = energy
                            best_feasible_ordering = ordering
                    else:
                        if energy < best_infeasible_energy:
                            best_infeasible_sample = sample
                            best_infeasible_energy = energy
                
                best_energy = None
                best_feasible = None
                best_ordering = None
                
                if best_feasible_sample is not None:
                    best_energy = best_feasible_energy
                    best_ordering = best_feasible_ordering
                    best_feasible = True
                elif best_infeasible_sample is not None:
                    best_energy = best_infeasible_energy
                    best_feasible = False
                
                minla_cost = minla.calculate_min_linear_arrangement(G, best_ordering) if best_feasible else None
                
                rel_gap = (minla_cost - optimal_cost) / optimal_cost if (best_feasible and optimal_cost) else None

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
                    'optimal_cost': optimal_cost,
                    'relative_gap': rel_gap,
                    'time_s': round(elapsed, 3),
                }

                all_results.append(row)
                processed = config_count

                print_result(config_count * num_seeds, total_configs * num_seeds, norm, space_type, annealing_type, beta_min, beta_max, best_feasible, best_energy, minla_cost, optimal_cost, elapsed, seed)
                
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
        best = df[df['feasible'] == 1].nsmallest(1, 'energy').iloc[0]
        print(f"\nLowest energy feasible solution:")
        print(f"  Space type: {best['space_type']}")
        print(f"  Annealing type: {best['annealing_type']}")
        print(f"  Beta range: ({best['beta_min']}, {best['beta_max']})")
        print(f"  Energy: {best['energy']:.2f}")
        print(f"  MinLA cost: {best['minla_cost']}")
    print(f"{'='*70}")
    
    return df

if __name__ == "__main__":
    df = run_experiment()