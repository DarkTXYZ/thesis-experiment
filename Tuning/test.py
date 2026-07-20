import os
import sys
import pickle
import time
import pandas as pd
import networkx as nx
import openjij as oj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results/tuning_experiment")

# Single accumulating master file - every run reads this, skips configs already
# in it, and overwrites it with old + newly run rows. Avoids the CSVs ballooning
# from re-reading previous cumulative snapshots on each run.
DETAILED_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_openjij_detailed.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_openjij_summary.csv")

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10

NUM_SWEEPS_GRID = [100, 500, 1000]

# OpenJijSA: annealed beta_min -> beta_max schedule.
BETA_MIN_GRID = [1e-9, 0.01, 0.1, 0.5]
BETA_MAX_GRID = [0.1, 0.3, 0.5, 1, 5, 10]
BETA_RANGE_GRID = [
    (beta_min, beta_max)
    for beta_min in BETA_MIN_GRID
    for beta_max in BETA_MAX_GRID
    if beta_min <= beta_max
]

# OpenJijSQA: beta is held FIXED all the way through (gamma is what anneals),
# so it does not behave like SA's beta_min/beta_max schedule and BETA_MAX_GRID's
# values don't transfer - calibrated empirically against this problem's actual
# QUBO coefficient scale (linear terms up to ~800, couplings ~15-35): beta
# below ~1 or above ~20 was near-always infeasible, gamma below ~1 or above
# ~60 likewise. Both grids stay inside the empirically productive region.
SQA_BETA_GRID = [1, 2, 3, 5, 10]
SQA_GAMMA_GRID = [1, 3, 10, 30]

# Trotter slices per logical qubit. OpenJijSQA-only: this is what actually
# turns on path-integral/quantum-tunneling behavior. openjij hard-errors below
# 2 ("trotter slices must be equal or larger than 2"), unlike dwave's
# PathIntegralAnnealingSampler where 1 is a valid ("no chains") value.
# OpenJijSA.sample() has no such parameter.
TROTTERS = [2, 4, 8]

SOLVERS = {
    "OpenJijSA": oj.SASampler(),
    "OpenJijSQA": oj.SQASampler(),
}

def load_existing_results():
    """Load the master detailed CSV so already-tested configs can be skipped."""
    if not os.path.exists(DETAILED_CSV):
        return pd.DataFrame(), set()

    existing_df = pd.read_csv(DETAILED_CSV)
    # Rows written before trotters was tracked ran with the sampler
    # default (1) - backfill so they keep matching on re-run and survive groupby.
    if "trotters" not in existing_df.columns:
        existing_df["trotters"] = 1
    else:
        existing_df["trotters"] = existing_df["trotters"].fillna(1).astype(int)

    # gamma (transverse field) only applies to OpenJijSQA; OpenJijSA rows have
    # no such concept, so they're pinned at 0 rather than left NaN (NaN keys
    # get silently dropped by groupby).
    if "gamma" not in existing_df.columns:
        existing_df["gamma"] = 0.0
    else:
        existing_df["gamma"] = existing_df["gamma"].fillna(0.0).astype(float)

    existing_keys = {
        (
            str(row.solver),
            int(row.graph_id),
            int(row.seed),
            int(row.num_sweeps),
            float(row.beta_min),
            float(row.beta_max),
            int(row.trotters),
            float(row.gamma),
        )
        for row in existing_df.itertuples()
    }
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
    for group in extra_data.values():
        for graph_data in group["graphs"]:
            graphs.append(graph_data)
    return graphs


def print_result(solver_name, config_count, total_configs, n, graph_id, seed,
                  num_sweeps, beta_range, trotters, gamma, feasible, cost, elapsed):
    status = "OK" if feasible else "--"
    cost_str = f"{cost:6.2f}" if cost is not None else "   N/A"
    print(
        f"[{solver_name}] [{config_count}/{total_configs}] N={n} graph={graph_id} seed={seed:<3} | "
        f"sweeps={num_sweeps:<5} beta=({beta_range[0]:.2e},{beta_range[1]:.2e}) gamma={gamma:<5.2g} "
        f"trotters={trotters:<3} | {status} cost={cost_str} | time={elapsed:.2f}s"
    )


def run_experiment():
    graphs = read_extra_graphs()

    base_configs = [
        (num_sweeps, beta_range)
        for num_sweeps in NUM_SWEEPS_GRID
        for beta_range in BETA_RANGE_GRID
    ]
    # trotters/gamma only mean something for OpenJijSQA; SA's configs stay
    # pinned at (1, 0.0) so both solvers share the same result schema.
    #
    # OpenJijSQA.sample() has no beta_min/beta_max schedule - it holds a single
    # fixed beta and anneals gamma (the transverse field) down instead, so
    # SA's BETA_RANGE_GRID has no equivalent there (see SQA_BETA_GRID /
    # SQA_GAMMA_GRID above). The fixed beta is stored as beta_min == beta_max
    # to keep the result schema aligned with SA's real (beta_min, beta_max) rows.
    solver_configs = {
        "OpenJijSA": [
            (num_sweeps, beta_range, 1, 0.0)
            for num_sweeps, beta_range in base_configs
        ],
        "OpenJijSQA": [
            (num_sweeps, (beta, beta), trotter, gamma)
            for num_sweeps in NUM_SWEEPS_GRID
            for beta in SQA_BETA_GRID
            for gamma in SQA_GAMMA_GRID
            for trotter in TROTTERS
        ],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing_df, existing_keys = load_existing_results()
    all_results = existing_df.to_dict("records") if not existing_df.empty else []
    if existing_keys:
        print(f"Found {len(existing_keys)} previously run (solver, graph, seed, config) combos - will skip them.")

    try:
        for solver_name, solver in SOLVERS.items():
            configs = solver_configs[solver_name]
            total_configs = len(configs)
            is_sqa = solver_name == "OpenJijSQA"

            for graph_data in graphs:
                G = convert_graph_data_to_nx(graph_data)
                n = G.number_of_nodes()
                m = G.number_of_edges()
                graph_id = graph_data["id"]
                lower_bound = graph_data["lower_bound"]
                bqm = minla.generate_bqm_instance(G)

                for config_count, (num_sweeps, beta_range, trotters, gamma) in enumerate(configs, 1):
                    for seed in SEEDS:
                        run_key = (
                            solver_name, int(graph_id), int(seed), int(num_sweeps),
                            float(beta_range[0]), float(beta_range[1]),
                            int(trotters), float(gamma),
                        )
                        if run_key in existing_keys:
                            continue

                        sample_kwargs = dict(
                            num_reads=NUM_READS,
                            num_sweeps=num_sweeps,
                            seed=seed,
                            sparse=True,
                        )
                        if is_sqa:
                            # beta_range is (beta, beta) here - see solver_configs.
                            sample_kwargs["beta"] = beta_range[1]
                            sample_kwargs["gamma"] = gamma
                            sample_kwargs["trotter"] = trotters
                        else:
                            sample_kwargs["beta_min"] = beta_range[0]
                            sample_kwargs["beta_max"] = beta_range[1]

                        t0 = time.time()
                        sampleset = solver.sample(bqm, **sample_kwargs)
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
                            "n": n,
                            "m": m,
                            "graph_id": graph_id,
                            "seed": seed,
                            "num_sweeps": num_sweeps,
                            "beta_min": beta_range[0],
                            "beta_max": beta_range[1],
                            "trotters": trotters,
                            "gamma": gamma,
                            "feasible": feasible,
                            "minla_cost": best_cost,
                            "lower_bound": lower_bound,
                            "approx_ratio": approx_ratio,
                            "time_s": round(elapsed, 3),
                        })

                        print_result(solver_name, config_count, total_configs, n, graph_id, seed,
                                     num_sweeps, beta_range, trotters, gamma, feasible, best_cost, elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved.")

    df = pd.DataFrame(all_results)
    df.to_csv(DETAILED_CSV, index=False)
    print(f"\nDetailed results saved to {DETAILED_CSV}")

    # Aggregate across graphs and seeds per solver + config
    agg_rows = []
    group_cols = ["solver", "num_sweeps", "beta_min", "beta_max", "trotters", "gamma"]
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

    # Best config per solver: highest feasibility rate, then lowest mean approx_ratio, then lowest runtime
    print("\n" + "=" * 70)
    print("BEST CONFIG PER SOLVER")
    print("=" * 70)
    for solver_name in SOLVERS:
        solver_df = agg_df[agg_df["solver"] == solver_name].copy()
        solver_df["mean_approx_ratio"] = solver_df["mean_approx_ratio"].fillna(float("inf"))
        solver_df = solver_df.sort_values(
            by=["feasibility_rate", "mean_approx_ratio", "mean_time_s"],
            ascending=[False, True, True],
        )
        best = solver_df.iloc[0]
        print(
            f"{solver_name}: num_sweeps={best['num_sweeps']}, "
            f"beta_range=({best['beta_min']:.2e}, {best['beta_max']:.2e}), "
            f"gamma={best['gamma']}, "
            f"trotters={best['trotters']} | "
            f"feasibility_rate={best['feasibility_rate']:.2%}, "
            f"mean_approx_ratio={best['mean_approx_ratio']:.4f}, "
            f"mean_time_s={best['mean_time_s']:.3f}"
        )

    return df, agg_df


if __name__ == "__main__":
    run_experiment()
