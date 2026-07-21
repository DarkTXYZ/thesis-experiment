import time
import networkx as nx
import openjij as oj
import optuna
import pandas as pd
from Baseline.lower_bound import MinLALowerBounds as minla_bound
from Utils.MinLA import generate_bqm_instance, decode_solution, calculate_min_linear_arrangement

optuna.logging.set_verbosity(optuna.logging.WARNING)

N = 25
GRAPH_SEED = 42

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10
NUM_SWEEPS = 1000
TRIALS = 100

BETA_BOUNDS = (1e-6, 1e6)
GAMMA_BOUNDS = (1e-6, 1e6)
TROTTER_BOUNDS = (2, 8)

# Infeasible trials still need a finite value to give Optuna - approx_ratio is
# always >=1.0 for a feasible config, so a fixed sentinel above any realistic
# ratio marks "infeasible" as strictly worse without using an unbounded value.
INFEASIBLE_APPROX_RATIO = 10.0

G = nx.erdos_renyi_graph(N, 0.5, seed=GRAPH_SEED)
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(N, 0.5, seed=GRAPH_SEED)

bqm = generate_bqm_instance(G)
bqm.normalize()

lower_bound = minla_bound(G).bound()

solver = oj.SQASampler()


def objective(trial):
    beta = trial.suggest_float("beta", *BETA_BOUNDS, log=True)
    gamma = trial.suggest_float("gamma", *GAMMA_BOUNDS, log=True)
    trotter = trial.suggest_int("trotter", *TROTTER_BOUNDS)

    t0 = time.time()
    best_cost = None
    feasible_seed_count = 0

    for seed in SEEDS:
        sampleset = solver.sample(
            bqm,
            num_reads=NUM_READS,
            num_sweeps=NUM_SWEEPS,
            beta=beta,
            gamma=gamma,
            trotter=trotter,
            seed=seed,
        )
        seed_feasible = False
        for sample in sampleset.samples():
            ordering, is_feasible = decode_solution(sample, N)
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

    print(
        f"trial {trial.number:<4} beta={beta:<8.3g} gamma={gamma:<8.3g} trotter={trotter:<3} "
        f"feas_rate={feasibility_rate:.0%} approx_ratio={approx_ratio:.4f} time={elapsed:.2f}s"
    )

    return feasibility_rate, approx_ratio


def run_search():
    print(f"N={N}, lower_bound={lower_bound}, {TRIALS} trials x {len(SEEDS)} seeds")

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=f"openjij_sqa_N{N}",
        
    )
    study.optimize(objective, n_trials=TRIALS)

    trials_df = study.trials_dataframe()
    csv_path = "test_one_instance_openjij_optuna_search.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\nPareto-optimal trials (feasibility_rate desc, approx_ratio asc):")
    for t in sorted(study.best_trials, key=lambda t: (-t.values[0], t.values[1])):
        print(f"  trial {t.number:<4} feas_rate={t.values[0]:.0%} approx_ratio={t.values[1]:.4f} params={t.params}")

    return study


if __name__ == "__main__":
    run_search()
