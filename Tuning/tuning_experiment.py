import os
import sys
import pickle
import time
import pandas as pd
import networkx as nx
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, PathIntegralAnnealingSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results/tuning_experiment")

DETAILED_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_detailed.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_summary.csv")

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10

PENALTY_METHODS = ["lucas", "exact"]
NUM_SWEEPS_GRID = [1, 5, 10, 50, 100, 500, 1000]
BETA_MIN_GRID = [1e-9, 0.01, 0.1, 0.5]
BETA_MAX_GRID = [0.1, 0.3, 0.5, 1, 5, 10]
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

def load_existing_results():
    if not os.path.exists(DETAILED_CSV):
        return pd.DataFrame(), set()

    existing_df = pd.read_csv(DETAILED_CSV)

    if "penalty" not in existing_df.columns:
        existing_df["penalty"] = "unknown"

    existing_keys = set()
    for row in existing_df.itertuples():
        # Older result files do not have dataset_key, so retain their
        # n/m/graph_id identity as a compatibility fallback.
        graph_identity = (
            str(row.dataset_key)
            if "dataset_key" in existing_df.columns
            and pd.notna(row.dataset_key)
            else (int(row.n), int(row.m), int(row.graph_id))
        )
        existing_keys.add((
            str(row.solver),
            str(row.penalty),
            graph_identity,
            int(row.seed),
            int(row.num_sweeps),
            float(row.beta_min),
            float(row.beta_max),
            str(row.beta_schedule_type),
        ))
    return existing_df, existing_keys


def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G


def read_extra_graphs():
    with open(DATASET_PATH, "rb") as f:
        extra_data = pickle.load(f)

    graphs = []
    for group_key, group in extra_data.items():
        for graph_data in group["graphs"]:
            graph = dict(graph_data)
            # graph_id is only unique within a group in quantum_extra.pkl.
            # Preserve it for reporting, but use the composite key for
            # resuming interrupted tuning runs.
            graph["dataset_key"] = (
                f"{group_key}:graph{graph_data['id']}"
            )
            graphs.append(graph)
    return graphs


def print_result(solver_name, penalty, config_count, total_configs, n, graph_id, seed,
                  num_sweeps, beta_range, schedule_type, feasible, cost, elapsed):
    status = "OK" if feasible else "--"
    cost_str = f"{cost:6.2f}" if cost is not None else "   N/A"
    print(
        f"[{solver_name}] [{penalty}] [{config_count}/{total_configs}] N={n} graph={graph_id} seed={seed:<3} | "
        f"sweeps={num_sweeps:<5} beta=({beta_range[0]:.2e},{beta_range[1]:.2e}) "
        f"type={schedule_type:9} | {status} cost={cost_str} | time={elapsed:.2f}s"
    )


def run_experiment():
    graphs = read_extra_graphs()

    configs = [
        (num_sweeps, beta_range, schedule_type)
        for num_sweeps in NUM_SWEEPS_GRID
        for beta_range in BETA_RANGE_GRID
        for schedule_type in SCHEDULE_TYPES
    ]
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing_df, existing_keys = load_existing_results()
    all_results = existing_df.to_dict("records") if not existing_df.empty else []
    if existing_keys:
        print(f"Found {len(existing_keys)} previously run (solver, penalty, graph, seed, config) combos - will skip them.")

    try:
        for solver_name, solver in SOLVERS.items():
            total_configs = len(configs)
            for graph_data in graphs:
                G = convert_graph_data_to_nx(graph_data)
                n = G.number_of_nodes()
                m = G.number_of_edges()
                graph_id = graph_data["id"]
                dataset_key = graph_data["dataset_key"]
                lower_bound = graph_data["lower_bound"]

                for penalty in PENALTY_METHODS:
                    bqm = minla.generate_bqm_instance(G, penalty_mode=penalty)

                    for config_count, (num_sweeps, beta_range, schedule_type) in enumerate(configs, 1):
                        for seed in SEEDS:
                            run_key = (
                                solver_name, penalty, dataset_key, int(seed), int(num_sweeps),
                                float(beta_range[0]), float(beta_range[1]), schedule_type,
                            )
                            legacy_run_key = (
                                solver_name, penalty, (n, m, int(graph_id)), int(seed),
                                int(num_sweeps), float(beta_range[0]), float(beta_range[1]),
                                schedule_type,
                            )
                            if run_key in existing_keys or legacy_run_key in existing_keys:
                                continue
                            
                            t0 = time.time()
                            
                            if solver_name == "PathIntegralAnnealingSampler":
                                
                                if schedule_type == "linear":
                                    Hp_field = np.linspace(beta_range[0], beta_range[1], num_sweeps)
                                    Hd_field = np.linspace(beta_range[1], beta_range[0], num_sweeps)
                                else:
                                    Hp_field = np.geomspace(beta_range[0], beta_range[1], num_sweeps)
                                    Hd_field = np.geomspace(beta_range[1], beta_range[0], num_sweeps)
                                
                                sampleset = solver.sample(
                                    bqm,
                                    num_reads=NUM_READS,
                                    num_sweeps=num_sweeps,
                                    beta_schedule_type='custom',
                                    Hp_field=Hp_field,
                                    Hd_field=Hd_field,
                                    seed=seed,
                                )
                            else:
                                sampleset = solver.sample(
                                    bqm,
                                    num_reads=NUM_READS,
                                    num_sweeps=num_sweeps,
                                    beta_range=list(beta_range),
                                    beta_schedule_type=schedule_type,
                                    seed=seed,
                                )
                            elapsed = time.time() - t0

                            best_cost = None
                            for sample in sampleset.samples():
                                ordering, is_feasible = minla.decode_solution(sample, n)
                                if is_feasible:
                                    cost = minla.calculate_min_linear_arrangement(G, ordering)
                                    if best_cost is None or cost < best_cost:
                                        best_cost = cost

                            feasible = best_cost is not None
                            approx_ratio = best_cost / lower_bound if feasible else None

                            all_results.append({
                                "solver": solver_name,
                                "penalty": penalty,
                                "n": n,
                                "m": m,
                                "graph_id": graph_id,
                                "dataset_key": dataset_key,
                                "seed": seed,
                                "num_sweeps": num_sweeps,
                                "beta_min": beta_range[0],
                                "beta_max": beta_range[1],
                                "beta_schedule_type": schedule_type,
                                "feasible": feasible,
                                "minla_cost": best_cost,
                                "lower_bound": lower_bound,
                                "approx_ratio": approx_ratio,
                                "time_s": round(elapsed, 3),
                            })

                            print_result(solver_name, penalty, config_count, total_configs, n, graph_id, seed,
                                         num_sweeps, beta_range, schedule_type, feasible, best_cost, elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved.")

    df = pd.DataFrame(all_results)
    df.to_csv(DETAILED_CSV, index=False)
    print(f"\nDetailed results saved to {DETAILED_CSV}")

    agg_rows = []
    # ADDED "penalty" TO THE GROUPING COLUMNS HERE
    group_cols = ["solver", "penalty", "num_sweeps", "beta_min", "beta_max", "beta_schedule_type"]
    
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

    print("\n" + "=" * 70)
    print("BEST CONFIG PER SOLVER AND PENALTY MODEL")
    print("=" * 70)
    
    # UPDATED THIS LOOP TO ITERATE OVER BOTH SOLVER AND PENALTY
    for solver_name in SOLVERS:
        for penalty in PENALTY_METHODS:
            solver_df = agg_df[(agg_df["solver"] == solver_name) & (agg_df["penalty"] == penalty)].copy()
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
                f"beta_range=({best['beta_min']:.2e}, {best['beta_max']:.2e}), "
                f"beta_schedule_type={best['beta_schedule_type']} | "
                f"feasibility_rate={best['feasibility_rate']:.2%}, "
                f"mean_approx_ratio={best['mean_approx_ratio']:.4f}, "
                f"mean_time_s={best['mean_time_s']:.3f}"
            )

    return df, agg_df


if __name__ == "__main__":
    run_experiment()