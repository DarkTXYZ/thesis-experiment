import os
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from dwave.samplers import PathIntegralAnnealingSampler, RandomSampler
import Utils.MinLA as minla
from Baseline.lower_bound import calculate_lower_obj_bound

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

def run_experiment():
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
            lower_bound = calculate_lower_obj_bound(G)

            # ============ PathIntegralAnnealingSampler ============
            print(f'\n  Graph {graph_id}:')
            print(f'    Running PathIntegralAnnealingSampler...')
            
            # Collect best feasible solutions from each seed
            best_feasible_costs_pim = []
            total_elapsed_pim = 0

            for seed in seeds:
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
            if feasible_pim:
                print(f'      PIM: Feasible={feasible_pim} | Avg Cost={avg_minla_cost_pim:.2f} | Approx Ratio={approx_ratio_pim:.4f} | Time={total_elapsed_pim:.2f}s')
            else:
                print(f'      PIM: Feasible={feasible_pim} | Time={total_elapsed_pim:.2f}s')
            
            if feasible_random:
                print(f'      Random: Feasible={feasible_random} | Avg Cost={avg_minla_cost_random:.2f} | Approx Ratio={approx_ratio_random:.4f} | Time={total_elapsed_random:.2f}s')
            else:
                print(f'      Random: Feasible={feasible_random} | Time={total_elapsed_random:.2f}s')
        
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
    run_experiment()
            
            