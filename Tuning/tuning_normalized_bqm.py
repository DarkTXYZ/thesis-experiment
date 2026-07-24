import os
import pickle
import sys
import time
import networkx as nx
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import BruteForceSampler
from dwave.samplers import SimulatedAnnealingSampler, PathIntegralAnnealingSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.MinLA import generate_bqm_instance, decode_solution, calculate_min_linear_arrangement, calculate_lower_obj_bound

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10

BETA_GRID = [float(b) for b in np.logspace(-2, 0, 3)]  # 1e-4 ... 1e0, one point per decade
# Only beta_min <= beta_max combos are valid schedules, so build that set up
# front rather than suggesting beta_min/beta_max independently and discarding
# the invalid half of the grid at trial time.
BETA_RANGES = [(lo, hi) for lo in BETA_GRID for hi in BETA_GRID if lo <= hi]
NUM_SWEEPS = 100
NUM_SWEEPS_PER_BETA = 1

# PathIntegralAnnealingSampler-only knobs; SimulatedAnnealingSampler.sample()
# has no such parameters at all. qubits_per_chain=1 collapses to plain
# classical SA (no chains); >1 is what actually turns on path-integral/
# quantum-tunneling behavior. Gamma is the transverse field magnitude and
# chain_coupler_strength ties a chain's Trotter slices together.
QUBITS_PER_CHAIN_BOUNDS = 1 
GAMMA_BOUNDS = [float(b) for b in np.logspace(-4, 4, 9)]
# GAMMA_BOUNDS = 1

SOLVERS = {
    # "SimulatedAnnealingSampler": SimulatedAnnealingSampler(),
    "PA": PathIntegralAnnealingSampler(),
}

# Infeasible trials still need a finite value to give Optuna - approx_ratio is
# always >=1.0 for a feasible config, so a fixed sentinel above any realistic
# ratio marks "infeasible" as strictly worse without using an unbounded value.
INFEASIBLE_APPROX_RATIO = 10.0

CSV_PATH = os.path.join(PARENT_DIR, "normalize_bqm_optuna_search.csv")


def build_beta_schedule(beta_min, beta_max, schedule_type):
    """Construct the per-sweep beta values ourselves instead of letting the
    sampler's 'linear'/'geometric' beta_schedule_type interpolate them, so we
    always pass beta_schedule_type='custom' downstream."""
    num_betas, rem = divmod(NUM_SWEEPS, NUM_SWEEPS_PER_BETA)
    if rem:
        raise ValueError("NUM_SWEEPS must be a multiple of NUM_SWEEPS_PER_BETA")

    if schedule_type == "linear":
        return np.linspace(beta_min, beta_max, num=num_betas)
    elif schedule_type == "geometric":
        return np.geomspace(beta_min, beta_max, num=num_betas)
    raise ValueError(f"Unknown beta schedule type: {schedule_type}")


def build_sample_kwargs(trial, solver_name):
    # suggest_categorical requires None/bool/int/float/str choices for
    # persistent storage, so suggest an index into BETA_RANGES rather than
    # the (lo, hi) tuple itself.
    beta_range_idx = trial.suggest_categorical("beta_range_idx", list(range(len(BETA_RANGES))))
    beta_min, beta_max = BETA_RANGES[beta_range_idx]
    schedule_type = trial.suggest_categorical("beta_schedule_type", ["linear", "geometric"])
    beta_schedule = build_beta_schedule(beta_min, beta_max, schedule_type)

    sample_kwargs = dict(
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        num_sweeps_per_beta=NUM_SWEEPS_PER_BETA,
        beta_schedule_type="custom",
    )

    if solver_name == "SA":
        sample_kwargs["beta_schedule"] = beta_schedule
        sample_kwargs["randomize_order"] = trial.suggest_categorical("randomize_order", [False, True])
        sample_kwargs["proposal_acceptance_criteria"] = trial.suggest_categorical(
            "proposal_acceptance_criteria", ["Metropolis", "Gibbs"])
    else:
        sample_kwargs["Hp_field"] = beta_schedule
        sample_kwargs["Hd_field"] = beta_schedule[::-1]
        sample_kwargs["qubits_per_chain"] = QUBITS_PER_CHAIN_BOUNDS
        # sample_kwargs["Gamma"] = GAMMA_BOUNDS

        # sample_kwargs["qubits_per_chain"] = trial.suggest_int("qubits_per_chain", *QUBITS_PER_CHAIN_BOUNDS)
        sample_kwargs["Gamma"] = trial.suggest_categorical("gamma", GAMMA_BOUNDS)
        # sample_kwargs["chain_coupler_strength"] = trial.suggest_float(
        #     "chain_coupler_strength", *CHAIN_COUPLER_STRENGTH_BOUNDS, log=True)

    return sample_kwargs


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


def load_graph_cache():
    """Precompute (G, n, lower_bound, bqm) once per graph so every trial reuses it."""
    cache = []
    for graph_data in read_extra_graphs():
        G = convert_graph_data_to_nx(graph_data)
        bqm = generate_bqm_instance(G)
        bqm.normalize()
        cache.append({
            "graph_id": graph_data["id"],
            "G": G,
            "n": G.number_of_nodes(),
            "lower_bound": graph_data["lower_bound"],
            "bqm": bqm,
        })
    return cache


def trial_config_key(trial, beta_min, beta_max):
    """Identify a trial by its resolved param *values* rather than
    beta_range_idx, since that index's meaning shifts whenever BETA_GRID
    changes. Assumes graph_cache/SEEDS/NUM_READS are unchanged between runs
    being compared, since those affect the result for a given config too."""
    other_params = tuple(
        sorted((k, v) for k, v in trial.params.items() if k not in ("beta_range_idx", "beta_schedule_type"))
    )
    return (round(beta_min, 12), round(beta_max, 12), trial.params["beta_schedule_type"]) + other_params


def load_cached_results(csv_path, solver_name):
    """Map config key -> prior COMPLETE result for solver_name, so reruns of
    the same grid can skip trials already recorded in csv_path."""
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    df = df[(df["solver"] == solver_name) & (df["state"] == "COMPLETE")]
    required = {"user_attrs_beta_min", "user_attrs_beta_max", "params_beta_schedule_type", "values_0", "values_1", "user_attrs_graphs_feasible"}
    if df.empty or not required.issubset(df.columns):
        return {}

    other_param_cols = sorted(
        c for c in df.columns
        if c.startswith("params_") and c not in ("params_beta_range_idx", "params_beta_schedule_type")
    )

    cached = {}
    for _, row in df.iterrows():
        key = (
            (round(row["user_attrs_beta_min"], 12), round(row["user_attrs_beta_max"], 12), row["params_beta_schedule_type"])
            + tuple(sorted((c[len("params_"):], row[c]) for c in other_param_cols))
        )
        cached[key] = {
            "graphs_feasible_rate": row["values_0"],
            "mean_approx_ratio": row["values_1"],
            "graphs_feasible": row["user_attrs_graphs_feasible"],
        }
    return cached


def make_objective(solver_name, solver, graph_cache, cached_results):
    """Evaluate one param combo across every graph, so the winner is the combo that
    stays feasible on the most graphs rather than one overfit to a single graph."""
    def objective(trial):
        sample_kwargs = build_sample_kwargs(trial, solver_name)
        beta_min, beta_max = BETA_RANGES[trial.params["beta_range_idx"]]
        schedule_type = trial.params["beta_schedule_type"]
        trial.set_user_attr("beta_min", beta_min)
        trial.set_user_attr("beta_max", beta_max)

        cached = cached_results.get(trial_config_key(trial, beta_min, beta_max))
        if cached is not None:
            trial.set_user_attr("graphs_feasible", cached["graphs_feasible"])
            trial.set_user_attr("time_s", 0.0)
            trial.set_user_attr("skipped", True)
            print(
                f"[{solver_name}] trial {trial.number:<4} SKIPPED (already in csv) "
                f"beta=({beta_min:.2e},{beta_max:.2e}) type={schedule_type:<9} "
                f"graphs_feasible={cached['graphs_feasible']}/{len(graph_cache)} "
                f"mean_approx_ratio={cached['mean_approx_ratio']:.4f}"
            )
            return cached["graphs_feasible_rate"], cached["mean_approx_ratio"]

        t0 = time.time()
        graphs_feasible = 0
        approx_ratios = []

        for graph in graph_cache:
            G, n, lower_bound, bqm = graph["G"], graph["n"], graph["lower_bound"], graph["bqm"]
            best_cost = None
            for seed in SEEDS:
                sampleset = solver.sample(bqm, seed=seed, **sample_kwargs)
                for sample in sampleset.samples():
                    ordering, is_feasible = decode_solution(sample, n)
                    if is_feasible:
                        cost = calculate_min_linear_arrangement(G, ordering)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
            if best_cost is not None:
                graphs_feasible += 1
                approx_ratios.append(best_cost / lower_bound)

        elapsed = time.time() - t0
        graphs_feasible_rate = graphs_feasible / len(graph_cache)
        mean_approx_ratio = sum(approx_ratios) / len(approx_ratios) if approx_ratios else INFEASIBLE_APPROX_RATIO

        trial.set_user_attr("graphs_feasible", graphs_feasible)
        trial.set_user_attr("time_s", round(elapsed, 3))

        print(
            f"[{solver_name}] trial {trial.number:<4} sweeps={sample_kwargs['num_sweeps']:<5} "
            f"beta=({beta_min:.2e},{beta_max:.2e}) type={schedule_type:<9} "
            f"gamma={sample_kwargs['Gamma']:<3} | "
            f"qpc={sample_kwargs['qubits_per_chain']:<3} | "
            f"graphs_feasible={graphs_feasible}/{len(graph_cache)} "
            f"mean_approx_ratio={mean_approx_ratio:.4f} time={elapsed:.2f}s"
        )

        return graphs_feasible_rate, mean_approx_ratio

    return objective


def run_search():
    graph_cache = load_graph_cache()
    grid_size = len(BETA_RANGES) * len(GAMMA_BOUNDS) * 2  # x2 for linear/geometric schedule_type

    print(f"{len(graph_cache)} graphs, {grid_size} grid points x {len(SEEDS)} seeds x {len(graph_cache)} graphs per solver")

    all_rows = []
    for solver_name, solver in SOLVERS.items():
        print(f"\n=== {solver_name} ===")
        cached_results = load_cached_results(CSV_PATH, solver_name)
        if cached_results:
            print(f"Found {len(cached_results)} already-completed {solver_name} configs in {CSV_PATH}; these will be skipped.")

        study = optuna.create_study(
            directions=["maximize", "minimize"],
            study_name=f"{solver_name}_robust",
            sampler=BruteForceSampler(),
        )
        study.optimize(
            make_objective(solver_name, solver, graph_cache, cached_results),
            n_jobs=8,
        )

        trials_df = study.trials_dataframe()
        trials_df["solver"] = solver_name
        all_rows.append(trials_df)

        print(f"\nPareto-optimal trials for {solver_name} (graphs_feasible desc, mean_approx_ratio asc):")
        for t in sorted(study.best_trials, key=lambda t: (-t.values[0], t.values[1])):
            print(f"  trial {t.number:<4} graphs_feasible={t.user_attrs['graphs_feasible']}/{len(graph_cache)} "
                  f"mean_approx_ratio={t.values[1]:.4f} params={t.params}")

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nResults saved to {CSV_PATH}")
    return df


if __name__ == "__main__":
    run_search()
