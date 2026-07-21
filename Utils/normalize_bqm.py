import os
import sys
import time
import networkx as nx
import optuna
import pandas as pd
from MinLA import generate_bqm_instance, decode_solution, calculate_min_linear_arrangement, calculate_lower_obj_bound
from dwave.samplers import SimulatedAnnealingSampler, PathIntegralAnnealingSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Baseline.lower_bound import MinLALowerBounds

optuna.logging.set_verbosity(optuna.logging.WARNING)

N = 25
GRAPH_SEED = 42

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10
TRIALS_PER_SOLVER = 100

# Search bounds shared by both samplers - both use the same beta-annealing
# schedule machinery. bqm.normalize() puts coefficients on a standard ~[-1,1]
# scale, so these are absolute beta values (no extra scale factor).
BETA_MIN_BOUNDS = 1.0
BETA_MAX_UPPER = 1e9
NUM_SWEEPS = 1000
NUM_SWEEPS_PER_BETA = 1

# PathIntegralAnnealingSampler-only knobs; SimulatedAnnealingSampler.sample()
# has no such parameters at all. qubits_per_chain=1 collapses to plain
# classical SA (no chains); >1 is what actually turns on path-integral/
# quantum-tunneling behavior. Gamma is the transverse field magnitude and
# chain_coupler_strength ties a chain's Trotter slices together.
QUBITS_PER_CHAIN_BOUNDS = (1, 4)
GAMMA_BOUNDS = (1e-9, 1e9)
CHAIN_COUPLER_STRENGTH_BOUNDS = (1e-9, 1e9)

SOLVERS = {
    # "SimulatedAnnealingSampler": SimulatedAnnealingSampler(),
    "PA": PathIntegralAnnealingSampler(),
}

# Infeasible trials still need a finite value to give Optuna - approx_ratio is
# always >=1.0 for a feasible config, so a fixed sentinel above any realistic
# ratio marks "infeasible" as strictly worse without using an unbounded value.
INFEASIBLE_APPROX_RATIO = 10.0


def build_sample_kwargs(trial, solver_name):
    # beta_min = trial.suggest_float("beta_min", *BETA_MIN_BOUNDS, log=True)
    beta_min = BETA_MIN_BOUNDS
    beta_max = trial.suggest_float("beta_max", BETA_MIN_BOUNDS, BETA_MAX_UPPER, log=True)
    schedule_type = trial.suggest_categorical("beta_schedule_type", ["linear", "geometric"])

    sample_kwargs = dict(
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        beta_range=[beta_min, beta_max],
        beta_schedule_type=schedule_type,
        num_sweeps_per_beta=NUM_SWEEPS_PER_BETA,
    )

    if solver_name == "SA":
        sample_kwargs["randomize_order"] = trial.suggest_categorical("randomize_order", [False, True])
        sample_kwargs["proposal_acceptance_criteria"] = trial.suggest_categorical(
            "proposal_acceptance_criteria", ["Metropolis", "Gibbs"])
    else:
        sample_kwargs["qubits_per_chain"] = trial.suggest_int("qubits_per_chain", *QUBITS_PER_CHAIN_BOUNDS)
        sample_kwargs["Gamma"] = trial.suggest_float("gamma", *GAMMA_BOUNDS, log=True)
        sample_kwargs["chain_coupler_strength"] = trial.suggest_float(
            "chain_coupler_strength", *CHAIN_COUPLER_STRENGTH_BOUNDS, log=True)

    return sample_kwargs


def make_objective(solver_name, solver, G, n, lower_bound, bqm):
    def objective(trial):
        sample_kwargs = build_sample_kwargs(trial, solver_name)

        t0 = time.time()
        best_cost = None
        feasible_seed_count = 0

        for seed in SEEDS:
            sampleset = solver.sample(bqm, seed=seed, **sample_kwargs)
            seed_feasible = False
            for sample in sampleset.samples():
                ordering, is_feasible = decode_solution(sample, n)
                if is_feasible:
                    seed_feasible = True
                    cost = calculate_min_linear_arrangement(G, ordering)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
            if seed_feasible:
                feasible_seed_count += 1

        elapsed = time.time() - t0
        feasibility_rate = feasible_seed_count / len(SEEDS)
        approx_ratio = best_cost / lower_bound if best_cost is not None else INFEASIBLE_APPROX_RATIO

        trial.set_user_attr("best_cost", best_cost)
        trial.set_user_attr("time_s", round(elapsed, 3))

        beta_min, beta_max = sample_kwargs["beta_range"]
        print(
            f"[{solver_name}] trial {trial.number:<4} sweeps={sample_kwargs['num_sweeps']:<5} "
            f"beta=({beta_min:.2e},{beta_max:.2e}) type={sample_kwargs['beta_schedule_type']:<9} "
            f"gamma={sample_kwargs['Gamma']:<3} | "
            f"ccs={sample_kwargs['chain_coupler_strength']:<3} | "
            f"qpc={sample_kwargs['qubits_per_chain']:<3} | "
            f"feas_rate={feasibility_rate:.0%} approx_ratio={approx_ratio:.4f} time={elapsed:.2f}s"
        )

        return feasibility_rate, approx_ratio

    return objective


def run_search():
    G = nx.erdos_renyi_graph(N, 0.5, seed=GRAPH_SEED)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N, 0.5, seed=GRAPH_SEED)
        
    n = G.number_of_nodes()
    lower_bound = MinLALowerBounds(G).bound()

    bqm = generate_bqm_instance(G)
    bqm.normalize()

    print(f"N={n}, lower_bound={lower_bound}, {TRIALS_PER_SOLVER} trials x {len(SEEDS)} seeds per solver")

    all_rows = []
    for solver_name, solver in SOLVERS.items():
        print(f"\n=== {solver_name} ===")
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            study_name=f"{solver_name}_N{n}",
        )
        study.optimize(
            make_objective(solver_name, solver, G, n, lower_bound, bqm),
            n_trials=TRIALS_PER_SOLVER,
            n_jobs=-1,
        )

        trials_df = study.trials_dataframe()
        trials_df["solver"] = solver_name
        all_rows.append(trials_df)

        print(f"\nPareto-optimal trials for {solver_name} (feasibility_rate desc, approx_ratio asc):")
        for t in sorted(study.best_trials, key=lambda t: (-t.values[0], t.values[1])):
            print(f"  trial {t.number:<4} feas_rate={t.values[0]:.0%} approx_ratio={t.values[1]:.4f} params={t.params}")

    df = pd.concat(all_rows, ignore_index=True)
    csv_path = "normalize_bqm_optuna_search.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


if __name__ == "__main__":
    run_search()
