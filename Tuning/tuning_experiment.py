import os
import sys
import pickle
import time
import pandas as pd
import networkx as nx
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, PathIntegralAnnealingSampler

# Ensure Utils is accessible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results/tuning_experiment")

DETAILED_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_detailed.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_summary.csv")

# Constants for Tuning
SEED = 42
NUM_READS = 25  # Increased reads to replace multiple seeds, better for statistical mean
NUM_SWEEPS = 500

PENALTY_METHODS = ["lucas", "exact"]
BETA_MIN_GRID = [1e-9, 0.01, 0.05, 0.1, 0.5]
BETA_MAX_GRID = [0.1, 0.5, 1, 5, 10]
BETA_RANGE_GRID = [
    (beta_min, beta_max)
    for beta_min in BETA_MIN_GRID
    for beta_max in BETA_MAX_GRID
    if beta_min <= beta_max
]
SCHEDULE_TYPES = ["linear", "geometric"]

SOLVERS = {
    "SimulatedAnnealingSampler": SimulatedAnnealingSampler(),
    "PathIntegralAnnealingSampler": PathIntegralAnnealingSampler(),
}

def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G


def read_extra_graphs():
    """Reads the tuning dataset and explicitly extracts the edge probability (p)."""
    with open(DATASET_PATH, "rb") as f:
        extra_data = pickle.load(f)

    graphs = []
    # Assuming extra_data is structured where group_key represents the density (e.g., 0.2, 0.5, 0.8)
    for group_key, group in extra_data.items():
        for graph_data in group["graphs"]:
            graph = dict(graph_data)
            graph["dataset_key"] = f"{group_key}:graph{graph_data['id']}"
            graph["p"] = group_key  # Track edge probability explicitly
            graphs.append(graph)
    return graphs


def print_result(solver_name, penalty, config_count, total_configs, n, p_val, graph_id, seed,
                  num_sweeps, beta_range, schedule_type, feasible, cost, elapsed):
    status = "OK" if feasible else "--"
    cost_str = f"{cost:6.2f}" if cost is not None else "   N/A"
    print(
        f"[{solver_name}] [{penalty}] [{config_count}/{total_configs}] N={n} p={p_val} graph={graph_id} seed={seed:<3} | "
        f"sweeps={num_sweeps:<5} beta=({beta_range[0]:.2e},{beta_range[1]:.2e}) "
        f"type={schedule_type:9} | {status} cost={cost_str} | time={elapsed:.2f}s"
    )


def run_experiment():
    graphs = read_extra_graphs()

    configs = [
        (beta_range, schedule_type)
        for beta_range in BETA_RANGE_GRID
        for schedule_type in SCHEDULE_TYPES
    ]
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    try:
        for solver_name, solver in SOLVERS.items():
            total_configs = len(configs)
            for graph_data in graphs:
                G = convert_graph_data_to_nx(graph_data)
                n = G.number_of_nodes()
                m = G.number_of_edges()
                graph_id = graph_data["id"]
                p_val = graph_data["p"]
                dataset_key = graph_data["dataset_key"]
                lower_bound = graph_data["lower_bound"]

                for penalty in PENALTY_METHODS:
                    bqm = minla.generate_bqm_instance(G, penalty_mode=penalty)

                    for config_count, (beta_range, schedule_type) in enumerate(configs, 1):
                        t0 = time.time()
                        
                        if solver_name == "PathIntegralAnnealingSampler":
                            if schedule_type == "linear":
                                Hp_field = np.linspace(beta_range[0], beta_range[1], NUM_SWEEPS)
                                Hd_field = np.linspace(beta_range[1], beta_range[0], NUM_SWEEPS)
                            else:
                                Hp_field = np.geomspace(beta_range[0], beta_range[1], NUM_SWEEPS)
                                Hd_field = np.geomspace(beta_range[1], beta_range[0], NUM_SWEEPS)

                            sampleset = solver.sample(
                                bqm,
                                num_reads=NUM_READS,
                                num_sweeps=NUM_SWEEPS,
                                beta_schedule_type='custom',
                                Hp_field=Hp_field,
                                Hd_field=Hd_field,
                                seed=SEED,
                            )
                        else:
                            sampleset = solver.sample(
                                bqm,
                                num_reads=NUM_READS,
                                num_sweeps=NUM_SWEEPS,
                                beta_range=list(beta_range),
                                beta_schedule_type=schedule_type,
                                seed=SEED,
                            )
                        elapsed = time.time() - t0

                        # --- RIGOROUS TUNING: Evaluate based on Mean Feasible Cost ---
                        feasible_costs = []
                        total_feasible_reads = 0

                        for datum in sampleset.data(["sample", "num_occurrences"]):
                            sample = datum.sample
                            occurrences = datum.num_occurrences
                            
                            ordering, is_feasible = minla.decode_solution(sample, n)
                            if is_feasible:
                                cost = minla.calculate_min_linear_arrangement(G, ordering)
                                feasible_costs.extend([cost] * occurrences)
                                total_feasible_reads += occurrences

                        feasible = len(feasible_costs) > 0
                        
                        if feasible:
                            best_cost = min(feasible_costs)
                            # Tune based on the expected (mean) cost of the sampler
                            metric_cost = np.mean(feasible_costs) 
                            approx_ratio = metric_cost / lower_bound
                        else:
                            best_cost = None
                            metric_cost = None
                            approx_ratio = None

                        all_results.append({
                            "solver": solver_name,
                            "penalty": penalty,
                            "p": p_val,
                            "n": n,
                            "m": m,
                            "graph_id": graph_id,
                            "dataset_key": dataset_key,
                            "seed": SEED,
                            "num_sweeps": NUM_SWEEPS,
                            "beta_min": beta_range[0],
                            "beta_max": beta_range[1],
                            "beta_schedule_type": schedule_type,
                            "feasible": feasible,
                            "minla_cost": metric_cost,  # Logging mean for tuning aggregation
                            "best_cost": best_cost,     # Logging best just for record keeping
                            "lower_bound": lower_bound,
                            "approx_ratio": approx_ratio,
                            "time_s": round(elapsed, 3),
                        })

                        print_result(solver_name, penalty, config_count, total_configs, n, p_val, graph_id, SEED,
                                     NUM_SWEEPS, beta_range, schedule_type, feasible, metric_cost, elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved.")

    # Save detailed data
    df = pd.DataFrame(all_results)
    df.to_csv(DETAILED_CSV, index=False)
    print(f"\nDetailed results saved to {DETAILED_CSV}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detailed_snapshot = os.path.join(RESULTS_DIR, f"tuning_experiment_detailed_{timestamp}.csv")
    df.to_csv(detailed_snapshot, index=False)
    print(f"Detailed results snapshot saved to {detailed_snapshot}")

    # Generate Summary Aggregation (Grouped by p)
    agg_rows = []
    group_cols = ["solver", "penalty", "p", "num_sweeps", "beta_min", "beta_max", "beta_schedule_type"]
    
    for keys, group in df.groupby(group_cols):
        feasible_runs = group[group["feasible"] == True]
        agg_rows.append({
            **dict(zip(group_cols, keys)),
            "feasibility_rate": len(feasible_runs) / len(group),
            "mean_approx_ratio": feasible_runs["approx_ratio"].mean() if len(feasible_runs) > 0 else None,
            "mean_time_s": group["time_s"].mean(),
            "num_runs": len(group),
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary results saved to {SUMMARY_CSV}")

    summary_snapshot = os.path.join(RESULTS_DIR, f"tuning_experiment_summary_{timestamp}.csv")
    agg_df.to_csv(summary_snapshot, index=False)
    print(f"Summary results snapshot saved to {summary_snapshot}")

    # Print the winning configurations per density
    print("\n" + "=" * 70)
    print("BEST CONFIG PER SOLVER, PENALTY MODEL, AND GRAPH DENSITY (p)")
    print("=" * 70)
    
    unique_ps = sorted([p for p in df["p"].unique() if pd.notna(p)])
    
    for p_val in unique_ps:
        print(f"\n--- Best Configurations for p = {p_val} ---")
        for solver_name in SOLVERS:
            for penalty in PENALTY_METHODS:
                solver_df = agg_df[
                    (agg_df["solver"] == solver_name) & 
                    (agg_df["penalty"] == penalty) & 
                    (agg_df["p"] == p_val)
                ].copy()
                
                if solver_df.empty:
                    continue
                
                solver_df["mean_approx_ratio"] = solver_df["mean_approx_ratio"].fillna(float("inf"))
                solver_df = solver_df.sort_values(
                    by=["feasibility_rate", "mean_approx_ratio", "mean_time_s"],
                    ascending=[False, True, True],
                )
                best = solver_df.iloc[0]
                print(
                    f"{solver_name} [{penalty.upper()}]: num_sweeps={best['num_sweeps']}, "
                    f"beta=({best['beta_min']:.2e}, {best['beta_max']:.2e}), "
                    f"type={best['beta_schedule_type']} | "
                    f"feas={best['feasibility_rate']:.2%}, "
                    f"gap={best['mean_approx_ratio']:.4f}, "
                    f"time={best['mean_time_s']:.3f}s"
                )

    return df, agg_df

if __name__ == "__main__":
    run_experiment()