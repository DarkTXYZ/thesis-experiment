import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler
from typing import Dict, Tuple, List
from collections import defaultdict
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results/tuning_experiment")
SEEDS = [42, 123, 456, 789, 999]  # 5 different seeds
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
RESULTS_CSV = os.path.join(RESULTS_DIR, f"PIM_test_default_beta_range_{TIMESTAMP}.csv")

def default_ising_beta_range(h, J,
                              max_single_qubit_excitation_rate = 0.01,
                              scale_T_with_N = True):
    if not 0 < max_single_qubit_excitation_rate < 1:
        raise ValueError('Targeted single qubit excitations rates must be in range (0,1)')

    sum_abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
    if sum_abs_bias_dict:
        min_abs_bias_dict = {k: v for k, v in sum_abs_bias_dict.items() if v != 0}
    else:
        min_abs_bias_dict = {}
    for (k1, k2), v in J.items():
        for k in [k1,k2]:
            sum_abs_bias_dict[k] += abs(v)
            if v != 0: 
                if k in min_abs_bias_dict:
                    min_abs_bias_dict[k] = min(abs(v),min_abs_bias_dict[k])
                else:
                    min_abs_bias_dict[k] = abs(v)

    if not min_abs_bias_dict:
        warn_msg = ('All bqm biases are zero (all energies are zero), this is '
                    'likely a value error. Temperature range is set arbitrarily '
                    'to [0.1,1]. Metropolis-Hastings update is non-ergodic.')
        warnings.warn(warn_msg)
        return([0.1,1])


    max_effective_field = max(sum_abs_bias_dict.values(), default=0)

    if max_effective_field == 0:
        hot_beta = 1
    else:
        hot_beta = np.log(2) / (2*max_effective_field)

    if len(min_abs_bias_dict)==0:
        cold_beta = hot_beta
    else:
        values_array = np.array(list(min_abs_bias_dict.values()),dtype=float)
        min_effective_field = np.min(values_array)
        if scale_T_with_N:
            number_min_gaps = np.sum(min_effective_field == values_array)
        else:
            number_min_gaps = 1
        cold_beta = np.log(number_min_gaps/max_single_qubit_excitation_rate) / (2*min_effective_field)

    return [hot_beta, cold_beta]

def default_beta_range(bqm):
    ising = bqm.spin
    return default_ising_beta_range(ising.linear, ising.quadratic)

def read_dataset():
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data
    return datasets

def decode_solution(raw_sample: Dict, n: int) -> Tuple[np.ndarray, bool]:
    sol = np.zeros((n, n), dtype=int)
    for u in range(n):
        for k in range(n):
            val = raw_sample.get(f'X[{u}][{k}]', 0)
            if val:
                sol[u, k] = 1
    is_feasible = check_feasibility(sol, n)
    ordering = np.sum(sol, axis=1)
    return ordering, is_feasible


def check_feasibility(sol: np.ndarray, n: int) -> bool:
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return False
    labels = np.sum(sol, axis=1)
    return len(np.unique(labels)) == n and np.all(labels > 0) and np.all(labels <= n)


def generate_field(space_type: str, beta_min: float, beta_max: float, num_sweeps: int) -> Tuple[np.ndarray, np.ndarray]:
    if space_type == 'linear':
        Hp_field = np.linspace(beta_min, beta_max, num=num_sweeps)
        Hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)
    elif space_type == 'geometric':
        Hp_field = np.geomspace(beta_min, beta_max, num=num_sweeps)
        Hd_field = np.geomspace(beta_max, beta_min, num=num_sweeps)
    else:
        Hp_field = np.logspace(beta_min, beta_max, num=num_sweeps)
        Hd_field = np.logspace(beta_max, beta_min, num=num_sweeps)
    return Hp_field, Hd_field


def print_result(config_count: int, total_configs: int, normalized: bool, space_type: str,
                 feasible: bool, energy: float, minla_cost, optimal_cost, elapsed: float, seed: int):
    status = "✓" if feasible else "✗"
    print(f"[{config_count}/{total_configs}] ({seed}) normalized={normalized} | {space_type:9} | {status} "
          f"E={energy:12.2f} | cost={minla_cost} | optimal_cost={optimal_cost} | {elapsed:.2f}s")


def run_experiment():
    datasets = read_dataset()
    
    # betas = np.array([1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
    # betas = np.array([5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
    # space_types = ['linear', 'geometric']
    space_types = ['linear', 'geometric']
    normalized = [False]
   
    graph_data = datasets[25]['graphs'][0]
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    solver = PathIntegralAnnealingSampler()
    num_sweeps = 1000
    num_reads = 10

    configs = []
    for norm in normalized:
        for space_type in space_types:
            configs.append((norm, space_type))
    
    total_configs = len(configs)
    num_seeds = len(SEEDS)
    all_results = []
    processed = 0

    try:
        for config_count, (norm, space_type) in enumerate(configs, 1):
            # Run each configuration with all 5 seeds and collect results
            seed_results = []
            
            for seed_idx, seed in enumerate(SEEDS, 1):
                bqm = minla.generate_bqm_instance(G)
                if norm:
                    bqm.normalize()
                optimal_cost = graph_data.get('optimal_cost', None)

                t0 = time.time()

                temp = default_beta_range(bqm)

                Hp_field, Hd_field = generate_field(space_type, temp[0], temp[1], num_sweeps)

                sampleset = solver.sample(
                    bqm,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    beta_schedule_type='custom',
                    seed=seed,
                    Hp_field=Hp_field,
                    Hd_field=Hd_field
                )

                elapsed = time.time() - t0

                # Find the best feasible solution from all 10 reads
                best_feasible = None
                best_feasible_energy = float('inf')
                best_infeasible = None
                best_infeasible_energy = float('inf')
                
                for sample, energy in zip(sampleset.samples(), sampleset.data_vectors['energy']):
                    ordering, feasible = decode_solution(sample, n)
                    if feasible:
                        if energy < best_feasible_energy:
                            best_feasible = sample
                            best_feasible_energy = energy
                    else:
                        if energy < best_infeasible_energy:
                            best_infeasible = sample
                            best_infeasible_energy = energy
                
                # Use best feasible if available, otherwise use best infeasible
                if best_feasible is not None:
                    energy = best_feasible_energy
                    ordering, feasible = decode_solution(best_feasible, n)
                elif best_infeasible is not None:
                    energy = best_infeasible_energy
                    ordering, feasible = decode_solution(best_infeasible, n)
                else:
                    # Fallback to first sample if nothing found
                    best = sampleset.first
                    energy = best.energy
                    ordering, feasible = decode_solution(best.sample, n)
                
                minla_cost = minla.calculate_min_linear_arrangement(G, ordering) if feasible else None
                rel_gap = (minla_cost - optimal_cost) / optimal_cost if (feasible and optimal_cost) else None

                seed_result = {
                    'n': n,
                    'm': m,
                    'bqm_is_normalized': norm,
                    'space_type': space_type,
                    'energy': energy,
                    'feasible': feasible,
                    'minla_cost': minla_cost,
                    'optimal_cost': optimal_cost,
                    'relative_gap': rel_gap,
                    'time_s': round(elapsed, 3),
                    'seed': seed,
                }
                seed_results.append(seed_result)
                
                print_result(config_count, total_configs, norm, space_type, feasible, energy, minla_cost, optimal_cost, elapsed, seed)
            
            # Select best result from all seeds (prefer feasible, then lower energy)
            best_result = max(seed_results, key=lambda x: (x['feasible'], -x['energy']))
            all_results.append(best_result)
            processed = config_count
            
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
        print(f"  Energy: {best['energy']:.2f}")
        print(f"  MinLA cost: {best['minla_cost']}")
    print(f"{'='*70}")
    
    return df


if __name__ == "__main__":
    df = run_experiment()