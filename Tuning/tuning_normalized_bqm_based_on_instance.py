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
from dwave.samplers.sa.sampler import default_beta_range

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.MinLA import generate_bqm_instance, decode_solution, calculate_min_linear_arrangement, calculate_lower_obj_bound

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")
CSV_PATH = os.path.join(PARENT_DIR, "normalize_bqm_based_on_instance_optuna_search.csv")

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10

# bqm.normalize() divides every bias by a single scale factor, which shrinks
# energy gaps by that same factor. Since acceptance probability goes as
# exp(-beta*deltaE), reaching the same physical "coldness" after normalizing
# requires beta to scale up by 1/scale_factor. That factor differs per graph
# (it depends on that graph's own max bias), so a single global beta grid
# shared across graphs is either too hot for some or absurdly slow for
# others. Instead we compute each graph's own hot/cold beta estimate (from
# its un-normalized biases, rescaled by its own normalization factor) and
# only tune a shared multiplier on top of that per-instance estimate.
BETA_SCALE_MULTIPLIERS = [0.1, 0.3, 1.0, 3.0, 10.0]
NUM_SWEEPS = 1000
NUM_SWEEPS_PER_BETA = 1

# PathIntegralAnnealingSampler-only knobs; SimulatedAnnealingSampler.sample()
# has no such parameters at all. qubits_per_chain=1 collapses to plain
# classical SA (no chains); >1 is what actually turns on path-integral/
# quantum-tunneling behavior. Gamma is the transverse field magnitude and
# chain_coupler_strength ties a chain's Trotter slices together.
QUBITS_PER_CHAIN_BOUNDS = 1
GAMMA_BOUNDS = [float(b) for b in np.logspace(-4, 4, 9)]

SOLVERS = {
    # "SimulatedAnnealingSampler": SimulatedAnnealingSampler(),
    "PA": PathIntegralAnnealingSampler(),
}

# Infeasible trials still need a finite value to give Optuna - approx_ratio is
# always >=1.0 for a feasible config, so a fixed sentinel above any realistic
# ratio marks "infeasible" as strictly worse without using an unbounded value.
INFEASIBLE_APPROX_RATIO = 10.0


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


def build_sample_kwargs(solver_name, beta_min, beta_max, schedule_type, gamma, randomize_order, proposal_acceptance_criteria):
    beta_schedule = build_beta_schedule(beta_min, beta_max, schedule_type)

    sample_kwargs = dict(
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        num_sweeps_per_beta=NUM_SWEEPS_PER_BETA,
        beta_schedule_type="custom",
    )

    if solver_name == "SA":
        sample_kwargs["beta_schedule"] = beta_schedule
        sample_kwargs["randomize_order"] = randomize_order
        sample_kwargs["proposal_acceptance_criteria"] = proposal_acceptance_criteria
    else:
        sample_kwargs["Hp_field"] = beta_schedule
        sample_kwargs["Hd_field"] = beta_schedule[::-1]
        sample_kwargs["qubits_per_chain"] = QUBITS_PER_CHAIN_BOUNDS
        sample_kwargs["Gamma"] = gamma

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
    """Precompute (G, n, lower_bound, bqm, beta_min_base, beta_max_base) once
    per graph so every trial reuses it. beta_{min,max}_base is that graph's
    own hot/cold beta estimate, computed from its un-normalized biases and
    rescaled by its own normalization factor - the per-instance equivalent
    of what default_beta_range() would recommend before normalizing."""
    cache = []
    for graph_data in read_extra_graphs():
        G = convert_graph_data_to_nx(graph_data)

        bqm_raw = generate_bqm_instance(G)
        bqm_raw.normalize()  # just to get the scale factor, not to keep the normalized BQM
        hot_beta, cold_beta = default_beta_range(bqm_raw)

        # bqm = generate_bqm_instance(G)
        # scale = bqm.normalize()

        cache.append({
            "graph_id": graph_data["id"],
            "G": G,
            "n": G.number_of_nodes(),
            "lower_bound": graph_data["lower_bound"],
            "bqm": bqm_raw,
            "beta_min_base": hot_beta,
            "beta_max_base": cold_beta,
        })
    return cache


def make_objective(solver_name, solver, graph_cache):
    """Evaluate one param combo across every graph, so the winner is the combo that
    stays feasible on the most graphs rather than one overfit to a single graph."""
    def objective(trial):
        beta_scale_multiplier = trial.suggest_categorical("beta_scale_multiplier", BETA_SCALE_MULTIPLIERS)
        schedule_type = trial.suggest_categorical("beta_schedule_type", ["linear", "geometric"])

        gamma = randomize_order = proposal_acceptance_criteria = None
        if solver_name == "SA":
            randomize_order = trial.suggest_categorical("randomize_order", [False, True])
            proposal_acceptance_criteria = trial.suggest_categorical(
                "proposal_acceptance_criteria", ["Metropolis", "Gibbs"])
        else:
            gamma = trial.suggest_categorical("gamma", GAMMA_BOUNDS)

        t0 = time.time()
        graphs_feasible = 0
        approx_ratios = []

        for graph in graph_cache:
            G, n, lower_bound, bqm = graph["G"], graph["n"], graph["lower_bound"], graph["bqm"]
            # Sorted as a safety guard: the hot<=cold ordering from
            # default_beta_range() isn't a hard guarantee for every instance.
            beta_min, beta_max = sorted((
                graph["beta_min_base"] * beta_scale_multiplier,
                graph["beta_max_base"] * beta_scale_multiplier,
            ))
            sample_kwargs = build_sample_kwargs(
                solver_name, beta_min, beta_max, schedule_type,
                gamma, randomize_order, proposal_acceptance_criteria,
            )

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
            f"[{solver_name}] trial {trial.number:<4} sweeps={NUM_SWEEPS:<5} "
            f"beta_scale={beta_scale_multiplier:<6} type={schedule_type:<9} "
            f"gamma={gamma} | "
            f"graphs_feasible={graphs_feasible}/{len(graph_cache)} "
            f"mean_approx_ratio={mean_approx_ratio:.4f} time={elapsed:.2f}s"
        )

        return graphs_feasible_rate, mean_approx_ratio

    return objective


def run_search():
    graph_cache = load_graph_cache()
    grid_size = len(BETA_SCALE_MULTIPLIERS) * len(GAMMA_BOUNDS) * 2  # x2 for linear/geometric schedule_type

    print(f"{len(graph_cache)} graphs, {grid_size} grid points x {len(SEEDS)} seeds x {len(graph_cache)} graphs per solver")
    for graph in graph_cache:
        print(f"  graph {graph['graph_id']}: beta_range_base=({graph['beta_min_base']:.3e}, {graph['beta_max_base']:.3e})")

    # all_rows = []
    # for solver_name, solver in SOLVERS.items():
    #     print(f"\n=== {solver_name} ===")
    #     study = optuna.create_study(
    #         directions=["maximize", "minimize"],
    #         study_name=f"{solver_name}_robust_based_on_instance",
    #         sampler=BruteForceSampler(),
    #     )
    #     study.optimize(
    #         make_objective(solver_name, solver, graph_cache),
    #         n_jobs=8,
    #     )

    #     trials_df = study.trials_dataframe()
    #     trials_df["solver"] = solver_name
    #     all_rows.append(trials_df)

    #     print(f"\nPareto-optimal trials for {solver_name} (graphs_feasible desc, mean_approx_ratio asc):")
    #     for t in sorted(study.best_trials, key=lambda t: (-t.values[0], t.values[1])):
    #         print(f"  trial {t.number:<4} graphs_feasible={t.user_attrs['graphs_feasible']}/{len(graph_cache)} "
    #               f"mean_approx_ratio={t.values[1]:.4f} params={t.params}")

    # df = pd.concat(all_rows, ignore_index=True)
    # df.to_csv(CSV_PATH, index=False)
    # print(f"\nResults saved to {CSV_PATH}")
    # return df


if __name__ == "__main__":
    run_search()
