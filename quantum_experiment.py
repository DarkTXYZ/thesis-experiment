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

def run_random_sampler_baseline(G, bqm, seeds, num_reads=10, display_mode="detailed"):
    """Run RandomSampler as a baseline for comparison."""
    n = G.number_of_nodes()
    best_feasible_costs = []
    total_elapsed = 0
    feasible_seed_count = 0

    solver = RandomSampler()

    for seed in seeds:
        t0 = time.time()

        sampleset = solver.sample(bqm, num_reads=num_reads, seed=seed)

        elapsed = time.time() - t0
        total_elapsed += elapsed

        best_feasible_cost = None
        feasible_count = 0

        for sample in sampleset.samples():
            ordering, feasible = minla.decode_solution(sample, n)

            if feasible:
                feasible_count += 1
                minla_cost = minla.calculate_min_linear_arrangement(G, ordering)

                if best_feasible_cost is None or minla_cost < best_feasible_cost:
                    best_feasible_cost = minla_cost

        if best_feasible_cost is not None:
            best_feasible_costs.append(best_feasible_cost)
            feasible_seed_count += 1

        if display_mode == "detailed":
            print_seed_progress(
                solver_name="RandomSampler",
                seed=seed,
                feasible_count=feasible_count,
                num_reads=num_reads,
                best_cost=best_feasible_cost,
                elapsed=elapsed,
            )

    if best_feasible_costs:
        avg_minla_cost = np.mean(best_feasible_costs)
        feasible = True
    else:
        avg_minla_cost = None
        feasible = False

    return feasible, avg_minla_cost, total_elapsed, feasible_seed_count

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

def format_optional(value, digits=4):
    """Format a numeric value or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def print_section(title, char="="):
    print("\n" + char * 100)
    print(title)
    print(char * 100)


def print_dataset_summary(datasets, expected_sizes):
    print_section("DATASET VERIFICATION")

    for vertex_count, expected in expected_sizes.items():
        actual_count = len(datasets[vertex_count]["graphs"])

        if actual_count == expected:
            status = "OK"
        else:
            status = "WARNING"

        print(
            f"[{status}] N={vertex_count:<2} | "
            f"graphs found = {actual_count:<3} | "
            f"expected = {expected:<3}"
        )


def print_graph_start(vertex_count, graph_id, total_graphs, G, lower_bound):
    print(
        f"\n[N={vertex_count}] Graph {graph_id + 1}/{total_graphs} "
        f"(graph_id={graph_id}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
        f"LB={lower_bound})"
    )


def print_seed_progress(solver_name, seed, feasible_count, num_reads, best_cost, elapsed):
    feasible = feasible_count > 0
    status = "FEASIBLE" if feasible else "INFEASIBLE"
    cost_text = f"{best_cost:.2f}" if best_cost is not None else "--"

    print(
        f"    [{solver_name}] seed={seed:<3} | "
        f"{status:<10} | "
        f"feasible samples={feasible_count}/{num_reads} | "
        f"best cost={cost_text:<8} | "
        f"time={elapsed:.2f}s"
    )


def print_graph_solver_summary(solver_name, feasible, avg_cost, ratio, total_time, feasible_seed_count, num_seeds):
    status = "FEASIBLE" if feasible else "INFEASIBLE"
    cost_text = f"{avg_cost:.2f}" if avg_cost is not None else "--"
    ratio_text = f"{ratio:.4f}" if ratio is not None else "--"

    print(
        f"    [{solver_name} summary] "
        f"{status} | "
        f"feasible seeds={feasible_seed_count}/{num_seeds} | "
        f"avg cost={cost_text} | "
        f"rho_UB={ratio_text} | "
        f"total time={total_time:.2f}s"
    )


def print_size_summary(vertex_count, solver_name, feasibility_rate, mean_ratio, std_ratio, feasible_count, total_graphs, total_time):
    print(f"  {solver_name}:")
    print(f"    - Feasible instances : {feasible_count}/{total_graphs}")
    print(f"    - Feasibility rate   : {feasibility_rate:.2%}")
    print(f"    - Mean rho_UB        : {format_optional(mean_ratio, 4)}")
    print(f"    - Std dev rho_UB     : {format_optional(std_ratio, 4)}")
    print(f"    - Total runtime      : {total_time:.2f}s")
    
def print_graph_minimal_progress(
    vertex_count,
    graph_id,
    total_graphs,
    feasible_pim,
    feasible_seed_count_pim,
    approx_ratio_pim,
    total_elapsed_pim,
    feasible_random=None,
    feasible_seed_count_random=None,
    approx_ratio_random=None,
    total_elapsed_random=None,
    skip_random=False,
):
    """Print one compact progress line per graph."""

    pim_status = "True" if feasible_pim else "False"
    pim_ratio = f"{approx_ratio_pim:.4f}" if approx_ratio_pim is not None else "--"

    msg = (
        f"[N={vertex_count}] "
        f"Graph {graph_id + 1}/{total_graphs} | "
        f"PIA: feasible={pim_status}, "
        f"seeds={feasible_seed_count_pim}/5, "
        f"rho_UB={pim_ratio}, "
        f"time={total_elapsed_pim:.2f}s"
    )

    if not skip_random:
        random_status = "True" if feasible_random else "False"
        random_ratio = f"{approx_ratio_random:.4f}" if approx_ratio_random is not None else "--"

        msg += (
            f" | RS: feasible={random_status}, "
            f"seeds={feasible_seed_count_random}/5, "
            f"rho_UB={random_ratio}, "
            f"time={total_elapsed_random:.2f}s"
        )

    print(msg)

def save_aggregate_summary(all_rows):
    """Generate aggregate summary statistics by vertex count and solver."""
    df = pd.DataFrame(all_rows)
    
    summary_rows = []
    for n in sorted(df['n'].unique()):
        for solver in ['PathIntegralAnnealingSampler', 'RandomSampler']:
            subset = df[(df['n'] == n) & (df['solver'] == solver)]
            
            total_graphs = len(subset)
            feasible_subset = subset[subset['feasible'] == True]
            num_feasible = len(feasible_subset)
            
            if num_feasible > 0:
                feasibility_rate = num_feasible / total_graphs
                mean_ratio = feasible_subset['approx_ratio'].mean()
                std_ratio = feasible_subset['approx_ratio'].std() if num_feasible >= 2 else None
            else:
                feasibility_rate = 0
                mean_ratio = None
                std_ratio = None
            
            total_time = subset['time_s'].sum()
            
            summary_rows.append({
                'N': n,
                'Solver': solver,
                'Total_Graphs': total_graphs,
                'Feasible_Instances': num_feasible,
                'Feasibility_Rate': feasibility_rate,
                'Mean_rho_UB': mean_ratio,
                'Std_rho_UB': std_ratio,
                'Total_Runtime_s': total_time
            })
    
    return pd.DataFrame(summary_rows)

def run_experiment(skip_random=False, display_mode="detailed"):
    if display_mode not in {"detailed", "minimal"}:
        raise ValueError("display_mode must be either 'detailed' or 'minimal'")
    
    datasets = read_dataset()

    vertices_count = [5, 10, 15, 20, 25]
    num_reads = 10
    num_sweeps = 1000
    seeds = [42, 123, 456, 789, 999]
    num_seeds = len(seeds)

    beta_min = 1e-9
    beta_max = 1

    expected_sizes = {5: 21, 10: 100, 15: 100, 20: 100, 25: 100}
    print_dataset_summary(datasets, expected_sizes)

    print_section("EXPERIMENT CONFIGURATION")
    print(f"Sampler                 : PathIntegralAnnealingSampler")
    print(f"Baseline                : {'Skipped' if skip_random else 'RandomSampler'}")
    print(f"num_reads               : {num_reads}")
    print(f"num_sweeps              : {num_sweeps}")
    print(f"Seeds                   : {seeds}")
    print(f"Selected beta schedule  : Linear")
    print(f"beta_min, beta_max      : {beta_min}, {beta_max}")
    print(f"Dataset sizes           : {vertices_count}")

    all_rows = []

    for vertex_count in vertices_count:
        graphs = datasets[vertex_count]["graphs"]
        total_graphs = len(graphs)

        print_section(f"RUNNING EXPERIMENT FOR N={vertex_count}", char="-")

        feasibility_cnt_pim = 0
        feasibility_cnt_random = 0
        approx_ratios_pim = []
        approx_ratios_random = []
        total_time_pim_size = 0
        total_time_random_size = 0

        for graph_id, graph in enumerate(graphs):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()

            bqm = minla.generate_bqm_instance(G)
            lower_bound = graph["lower_bound"]

            if display_mode == "detailed":
                print_graph_start(
                    vertex_count=vertex_count,
                    graph_id=graph_id,
                    total_graphs=total_graphs,
                    G=G,
                    lower_bound=lower_bound,
                )

            # =====================================================
            # PathIntegralAnnealingSampler
            # =====================================================
            best_feasible_costs_pim = []
            total_elapsed_pim = 0
            feasible_seed_count_pim = 0

            solver = PathIntegralAnnealingSampler()

            Hp_field = np.linspace(beta_min, beta_max, num=num_sweeps)
            Hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)

            for seed in seeds:
                t0 = time.time()

                sampleset = solver.sample(
                    bqm,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    beta_schedule_type="custom",
                    Hp_field=Hp_field,
                    Hd_field=Hd_field,
                    seed=seed,
                )

                elapsed = time.time() - t0
                total_elapsed_pim += elapsed

                best_feasible_cost = None
                feasible_count = 0

                for sample in sampleset.samples():
                    ordering, feasible = minla.decode_solution(sample, n)

                    if feasible:
                        feasible_count += 1
                        minla_cost = minla.calculate_min_linear_arrangement(G, ordering)

                        if best_feasible_cost is None or minla_cost < best_feasible_cost:
                            best_feasible_cost = minla_cost

                if best_feasible_cost is not None:
                    best_feasible_costs_pim.append(best_feasible_cost)
                    feasible_seed_count_pim += 1

                if display_mode == "detailed":
                    print_seed_progress(
                        solver_name="PathIntegralAnnealingSampler",
                        seed=seed,
                        feasible_count=feasible_count,
                        num_reads=num_reads,
                        best_cost=best_feasible_cost,
                        elapsed=elapsed,
                    )

            total_time_pim_size += total_elapsed_pim

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

            if display_mode == "detailed":
                print_graph_solver_summary(
                    solver_name="PathIntegralAnnealingSampler",
                    feasible=feasible_pim,
                    avg_cost=avg_minla_cost_pim,
                    ratio=approx_ratio_pim,
                    total_time=total_elapsed_pim,
                    feasible_seed_count=feasible_seed_count_pim,
                    num_seeds=num_seeds,
                )

            # =====================================================
            # RandomSampler Baseline
            # =====================================================
            if not skip_random:
                feasible_random, avg_minla_cost_random, total_elapsed_random, feasible_seed_count_random = (
                    run_random_sampler_baseline(
                        G=G,
                        bqm=bqm,
                        seeds=seeds,
                        num_reads=num_reads,
                        display_mode=display_mode,
                    )
                )

                total_time_random_size += total_elapsed_random

                if feasible_random:
                    feasibility_cnt_random += 1
                    approx_ratio_random = avg_minla_cost_random / lower_bound
                    approx_ratios_random.append(approx_ratio_random)
                else:
                    approx_ratio_random = None

                if display_mode == "detailed":
                    print_graph_solver_summary(
                        solver_name="RandomSampler",
                        feasible=feasible_random,
                        avg_cost=avg_minla_cost_random,
                        ratio=approx_ratio_random,
                        total_time=total_elapsed_random,
                        feasible_seed_count=feasible_seed_count_random,
                        num_seeds=num_seeds,
                    )

            else:
                feasible_random = False
                avg_minla_cost_random = None
                approx_ratio_random = None
                total_elapsed_random = 0
                feasible_seed_count_random = 0

            # =====================================================
            # Record Results
            # =====================================================
            row_pim = {
                "n": n,
                "m": m,
                "graph_id": graph_id,
                "solver": "PathIntegralAnnealingSampler",
                "feasible": feasible_pim,
                "feasible_seed_count": feasible_seed_count_pim,
                "avg_minla_cost": avg_minla_cost_pim,
                "lower_bound": lower_bound,
                "approx_ratio": approx_ratio_pim,
                "time_s": round(total_elapsed_pim, 3),
                "num_seeds": num_seeds,
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "beta_schedule": "linear",
            }
            all_rows.append(row_pim)

            row_random = {
                "n": n,
                "m": m,
                "graph_id": graph_id,
                "solver": "RandomSampler",
                "feasible": feasible_random,
                "feasible_seed_count": feasible_seed_count_random,
                "avg_minla_cost": avg_minla_cost_random,
                "lower_bound": lower_bound,
                "approx_ratio": approx_ratio_random,
                "time_s": round(total_elapsed_random, 3),
                "num_seeds": num_seeds,
                "num_reads": num_reads,
                "num_sweeps": None,
                "beta_min": None,
                "beta_max": None,
                "beta_schedule": None,
            }
            all_rows.append(row_random)

        # =====================================================
        # Size-level summary
        # =====================================================
        feasibility_rate_pim = feasibility_cnt_pim / total_graphs
        feasibility_rate_random = feasibility_cnt_random / total_graphs

        avg_approx_ratio_pim = np.mean(approx_ratios_pim) if approx_ratios_pim else None
        std_approx_ratio_pim = np.std(approx_ratios_pim, ddof=1) if len(approx_ratios_pim) >= 2 else None

        avg_approx_ratio_random = np.mean(approx_ratios_random) if approx_ratios_random else None
        std_approx_ratio_random = np.std(approx_ratios_random, ddof=1) if len(approx_ratios_random) >= 2 else None
        
        if display_mode == "minimal":
            print_graph_minimal_progress(
                vertex_count=vertex_count,
                graph_id=graph_id,
                total_graphs=total_graphs,
                feasible_pim=feasible_pim,
                feasible_seed_count_pim=feasible_seed_count_pim,
                approx_ratio_pim=approx_ratio_pim,
                total_elapsed_pim=total_elapsed_pim,
                feasible_random=feasible_random,
                feasible_seed_count_random=feasible_seed_count_random,
                approx_ratio_random=approx_ratio_random,
                total_elapsed_random=total_elapsed_random,
                skip_random=skip_random,
            )

        print_section(f"SUMMARY FOR N={vertex_count}", char="-")

        print_size_summary(
            vertex_count=vertex_count,
            solver_name="PathIntegralAnnealingSampler",
            feasibility_rate=feasibility_rate_pim,
            mean_ratio=avg_approx_ratio_pim,
            std_ratio=std_approx_ratio_pim,
            feasible_count=feasibility_cnt_pim,
            total_graphs=total_graphs,
            total_time=total_time_pim_size,
        )

        print_size_summary(
            vertex_count=vertex_count,
            solver_name="RandomSampler",
            feasibility_rate=feasibility_rate_random,
            mean_ratio=avg_approx_ratio_random,
            std_ratio=std_approx_ratio_random,
            feasible_count=feasibility_cnt_random,
            total_graphs=total_graphs,
            total_time=total_time_random_size,
        )

    # =====================================================
    # Save results
    # =====================================================
    df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    detailed_csv = os.path.join(
        RESULTS_DIR,
        f"quantum_experiment_detailed_{timestamp}.csv",
    )
    df.to_csv(detailed_csv, index=False)

    summary_df = save_aggregate_summary(all_rows)
    summary_csv = os.path.join(
        RESULTS_DIR,
        f"quantum_experiment_summary_{timestamp}.csv",
    )
    summary_df.to_csv(summary_csv, index=False)

    print_section("AGGREGATE RESULTS SUMMARY")
    print(summary_df.to_string(index=False))

    print_section("OUTPUT FILES")
    print(f"Detailed results saved to : {detailed_csv}")
    print(f"Summary results saved to  : {summary_csv}")
    
if __name__ == "__main__":
    skip_random = False

    # Options:
    # display_mode = "detailed"  # seed-level progress
    # display_mode = "minimal"   # one line per graph
    display_mode = "minimal"

    run_experiment(
        skip_random=skip_random,
        display_mode=display_mode,
    )
            
            