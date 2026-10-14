import os
import pickle
import re
import time
import numpy as np
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler
import Utils.MinLA as minla
import networkx as nx

DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_PATH = "Results/pia_experiment"

SEEDS = [42]
NUM_READS = 25
NUM_SWEEPS = 500

PENALTY_METHODS = ["lucas", "exact"]

# Best-performing hyperparameters from the tuning experiment (Table
# tab:tuning-results), one configuration per penalty method held fixed
# across all graph densities.
TUNED_CONFIGS = {
    "lucas": {
        "beta_schedule_type": "geometric",
        "beta_range": (0.5, 5.0)
    },
    "exact": {
        "beta_schedule_type": "geometric",
        "beta_range": (0.05, 10.0)
    }
}


def extract_density(filename):
    """Target edge probability encoded in the filename, or None for an
    exhaustively-enumerated file (e.g. quantum_n5.pkl) with no fixed density."""
    match = re.search(r"_d([0-9.]+)\.pkl$", filename)
    return match.group(1) if match else None

def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def print_result(run_idx, total_runs, cumulative_elapsed, filename, graph_id, total_graphs,
                  penalty, feasible, feasible_read_count, elapsed):
    status = "OK" if feasible else "--"
    pct = 100 * run_idx / total_runs
    avg_time = cumulative_elapsed / run_idx
    eta = avg_time * (total_runs - run_idx)
    print(
        f"[{run_idx}/{total_runs} {pct:5.1f}% ETA {format_duration(eta):>6}] "
        f"[{filename}] graph {graph_id + 1}/{total_graphs} | penalty={penalty.upper():5} | "
        f"{status} reads={feasible_read_count} | time={elapsed:.3f}s"
    )


def run_experiment():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    solver = PathIntegralAnnealingSampler()

    all_rows = []

    dataset_files = []
    for filename in sorted(os.listdir(DATASET_PATH)):
        if not filename.endswith(".pkl") or filename == "quantum_extra.pkl":
            continue
        with open(os.path.join(DATASET_PATH, filename), "rb") as f:
            dataset_files.append((filename, pickle.load(f)))

    total_runs = sum(len(data["graphs"]) for _, data in dataset_files) * len(PENALTY_METHODS)
    run_idx = 0
    cumulative_elapsed = 0.0

    print(f"seed={SEEDS[0]} num_sweeps={NUM_SWEEPS} num_reads={NUM_READS} | {total_runs} runs total\n")

    for filename, data in dataset_files:
        total_graphs = len(data["graphs"])
        density = extract_density(filename)
        print(f"[{filename}] N={data['num_vertices']} | {total_graphs} graphs")

        for graph_id, graph in enumerate(data["graphs"]):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            lower_bound = graph["lower_bound"]

            for penalty in PENALTY_METHODS:
                run_idx += 1
                config = TUNED_CONFIGS[penalty]
                bqm = minla.generate_bqm_instance(G, penalty_mode=penalty)

                feasible_costs = []
                total_elapsed = 0

                # Precompute the longitudinal and transverse fields for path-integral simulation
                b_min, b_max = config["beta_range"]
                if config["beta_schedule_type"] == "linear":
                    Hp_field = np.linspace(b_min, b_max, NUM_SWEEPS)
                    Hd_field = np.linspace(b_max, b_min, NUM_SWEEPS)
                else:
                    Hp_field = np.geomspace(b_min, b_max, NUM_SWEEPS)
                    Hd_field = np.geomspace(b_max, b_min, NUM_SWEEPS)

                for seed in SEEDS:
                    t0 = time.time()
                    sampleset = solver.sample(
                        bqm,
                        num_reads=NUM_READS,
                        num_sweeps=NUM_SWEEPS,
                        seed=seed,
                        beta_schedule_type="custom",
                        Hp_field=Hp_field,
                        Hd_field=Hd_field
                    )
                    elapsed = time.time() - t0
                    total_elapsed += elapsed

                    for sample in sampleset.samples():
                        ordering, is_feasible = minla.decode_solution(sample, n)
                        if is_feasible:
                            cost = minla.calculate_min_linear_arrangement(G, ordering)
                            feasible_costs.append(cost)

                feasible = len(feasible_costs) > 0
                feasible_read_count = len(feasible_costs)
                # Best feasible solution among the reads, not an average: the
                # average is undefined when few reads are feasible and would
                # unfairly penalize configurations with a lower feasibility rate.
                best_minla_cost = min(feasible_costs) if feasible else None
                approx_ratio = best_minla_cost / lower_bound if feasible else None

                all_rows.append({
                    "n": n,
                    "m": m,
                    "density": density,
                    "graph_id": graph_id,
                    "solver": "PathIntegralAnnealingSampler",
                    "penalty": penalty,
                    "feasible": feasible,
                    "feasible_read_count": feasible_read_count,
                    "best_minla_cost": best_minla_cost,
                    "lower_bound": lower_bound,
                    "approx_ratio": approx_ratio,
                    "time_s": round(total_elapsed, 3),
                    "num_seeds": len(SEEDS),
                    "num_reads": NUM_READS,
                    "num_sweeps": NUM_SWEEPS,
                    "beta_schedule_type": config["beta_schedule_type"],
                    "beta_min": config["beta_range"][0],
                    "beta_max": config["beta_range"][1]
                })

                cumulative_elapsed += total_elapsed
                print_result(run_idx, total_runs, cumulative_elapsed, filename, graph_id, total_graphs,
                             penalty, feasible, feasible_read_count, total_elapsed)

    df = pd.DataFrame(all_rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_PATH, f"pia_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary -> {csv_path}")


if __name__ == "__main__":
    run_experiment()