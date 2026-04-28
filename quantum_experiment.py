import os
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler, RandomSampler
import Utils.MinLA as minla
from Baseline.lower_bound import calculate_lower_obj_bound
from collections import defaultdict
import warnings


DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_DIR = "Results/quantum_experiment"

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

def run_random_sampler_baseline(G, bqm, seeds, num_reads=10):
    """Run RandomSampler as a baseline for comparison."""
    n = G.number_of_nodes()
    best_feasible_costs = []
    total_elapsed = 0
    
    solver = RandomSampler()
    
    for seed in seeds:
        t0 = time.time()
        
        sampleset = solver.sample(bqm, num_reads=num_reads, seed=seed)
        
        elapsed = time.time() - t0
        total_elapsed += elapsed
        
        # Find the best feasible solution from this seed
        best_feasible_cost = None
        for sample in sampleset.samples():
            ordering, feasible = minla.decode_solution(sample, n)
            if feasible:
                minla_cost = minla.calculate_min_linear_arrangement(G, ordering)
                if best_feasible_cost is None or minla_cost < best_feasible_cost:
                    best_feasible_cost = minla_cost
        
        if best_feasible_cost is not None:
            best_feasible_costs.append(best_feasible_cost)
    
    # Calculate average of best feasible solutions
    if best_feasible_costs:
        avg_minla_cost = np.mean(best_feasible_costs)
        feasible = True
    else:
        avg_minla_cost = None
        feasible = False
    
    return feasible, avg_minla_cost, total_elapsed

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

def run_experiment(skip_random=False):
    datasets = read_dataset()
    
    vertices_count = [5, 10, 15, 20, 25]
    num_sweeps = 1000
    seeds = [42, 123, 456, 789, 999]
    num_seeds = len(seeds)

    beta_min = 1e-9
    beta_max = 1
    
    all_rows = []

    for vertex_count in vertices_count:
        print(f"\nRunning experiment for graph with {vertex_count} vertices...")
        graphs = datasets[vertex_count]['graphs']

        feasibility_cnt_pim = 0
        feasibility_cnt_random = 0
        approx_ratios_pim = []
        approx_ratios_random = []

        for graph_id, graph in enumerate(graphs):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()

            bqm = minla.generate_bqm_instance(G)
            lower_bound = graph['lower_bound']  # Use lower bound from dataset

            # ============ PathIntegralAnnealingSampler ============
            # Collect best feasible solutions from each seed
            best_feasible_costs_pim = []
            total_elapsed_pim = 0

            for seed in seeds:
                t0 = time.time()
                
                solver = PathIntegralAnnealingSampler()

                # temp = default_beta_range(bqm)

                # beta_schedule_type = 'custom'
                # Hp_field = np.linspace(temp[0], temp[1], num=num_sweeps)
                # Hd_field = np.linspace(temp[1], temp[0], num=num_sweeps)

                sampleset = solver.sample(
                    bqm,
                    num_reads=10,
                    num_sweeps=num_sweeps,
                    # beta_schedule_type='custom',
                    # beta_schedule_type=beta_schedule_type,
                    # Hp_field=Hp_field,
                    # Hd_field=Hd_field,
                    seed=seed
                )
                
                elapsed = time.time() - t0
                total_elapsed_pim += elapsed

                # Find the best feasible solution from this seed
                best_feasible_cost = None
                best_energy = None
                for sample in sampleset.samples():
                    ordering, feasible = minla.decode_solution(sample, n)
                    if feasible:
                        minla_cost = minla.calculate_min_linear_arrangement(G, ordering)
                        if best_feasible_cost is None or minla_cost < best_feasible_cost:
                            best_feasible_cost = minla_cost
                            best_energy = sampleset.data_vectors['energy'][list(sampleset.samples()).index(sample)]
                
                if best_feasible_cost is not None:
                    best_feasible_costs_pim.append(best_feasible_cost)

            # Calculate average of best feasible solutions
            if best_feasible_costs_pim:
                avg_minla_cost_pim = np.mean(best_feasible_costs_pim)
                feasibility_cnt_pim += 1
                approx_ratio_pim = avg_minla_cost_pim / lower_bound
                approx_ratios_pim.append(approx_ratio_pim)
                feasible_pim = True
            else:
                avg_minla_cost_pim = None
                approx_ratio_pim = None
                feasible_pim = False

            # ============ RandomSampler Baseline ============
            if not skip_random:
                print(f'    Running RandomSampler baseline...')
                feasible_random, avg_minla_cost_random, total_elapsed_random = run_random_sampler_baseline(
                    G, bqm, seeds, num_reads=10
                )
                
                if feasible_random:
                    feasibility_cnt_random += 1
                    approx_ratio_random = avg_minla_cost_random / lower_bound
                    approx_ratios_random.append(approx_ratio_random)
                else:
                    approx_ratio_random = None
            else:
                feasible_random = False
                avg_minla_cost_random = None
                approx_ratio_random = None
                total_elapsed_random = 0

            # ============ Record Results ============
            # PIM Results
            row_pim = {
                'n': n,
                'm': m,
                'graph_id': graph_id,
                'solver': 'PathIntegralAnnealingSampler',
                'feasible': feasible_pim,
                'avg_minla_cost': avg_minla_cost_pim,
                'lower_bound': lower_bound,
                'approx_ratio': approx_ratio_pim,
                'time_s': round(total_elapsed_pim, 3),
                'num_seeds': num_seeds
            }
            all_rows.append(row_pim)
            
            # Random Sampler Results
            row_random = {
                'n': n,
                'm': m,
                'graph_id': graph_id,
                'solver': 'RandomSampler',
                'feasible': feasible_random,
                'avg_minla_cost': avg_minla_cost_random,
                'lower_bound': lower_bound,
                'approx_ratio': approx_ratio_random,
                'time_s': round(total_elapsed_random, 3),
                'num_seeds': num_seeds
            }
            all_rows.append(row_random)

            # Print Results
            pim_str = f"PIM: Feasible={feasible_pim}"
            if feasible_pim:
                pim_str += f" | Cost={avg_minla_cost_pim:.2f} | Ratio={approx_ratio_pim:.4f}"
            pim_str += f" | Time={total_elapsed_pim:.2f}s"
            
            random_str = ""
            if not skip_random:
                random_str = f" | Random: Feasible={feasible_random}"
                if feasible_random:
                    random_str += f" | Cost={avg_minla_cost_random:.2f} | Ratio={approx_ratio_random:.4f}"
                random_str += f" | Time={total_elapsed_random:.2f}s"
            
            print(f"      Graph {graph_id}: {pim_str}{random_str}")
        
        feasibility_rate_pim = feasibility_cnt_pim / len(graphs)
        feasibility_rate_random = feasibility_cnt_random / len(graphs)
        avg_approx_ratio_pim = sum(approx_ratios_pim) / len(approx_ratios_pim) if approx_ratios_pim else None
        avg_approx_ratio_random = sum(approx_ratios_random) / len(approx_ratios_random) if approx_ratios_random else None
        
        print(f'\n  ====== Summary for {vertex_count} vertices ======')
        print(f'  PIM - Feasibility rate: {feasibility_rate_pim:.2%} | Avg approx ratio: {avg_approx_ratio_pim}')
        print(f'  Random - Feasibility rate: {feasibility_rate_random:.2%} | Avg approx ratio: {avg_approx_ratio_random}')
    
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
    skip_random = True
    run_experiment(skip_random=skip_random)
            
            