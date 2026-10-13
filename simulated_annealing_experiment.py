import os
import pickle
import re
import time
import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler
import Utils.MinLA as minla
import networkx as nx

DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_PATH = "Results/sa_experiment"

SEEDS = [42]
NUM_READS = 50
NUM_SWEEPS = 1000

PENALTY_METHODS = ["lucas", "exact"]

# Optimal hyperparameters derived from the tuning experiment, keyed by graph
# density (qubits_per_chain ignored - SA only ever runs with the default, 1).
TUNED_CONFIGS = {
    "0.2": {
        "beta_schedule_type": "geometric",
        "beta_range": (0.01, 1.0)
    },
    "0.5": {
        "beta_schedule_type": "geometric",
        "beta_range": (0.1, 0.5)
    },
    "0.8": {
        "beta_schedule_type": "linear",
        "beta_range": (0.01, 0.5)
    }
}


def extract_density(filename):
    match = re.search(r"_d([0-9.]+)\.pkl$", filename)
    if not match:
        raise ValueError(f"Could not determine density from filename: {filename}")
    return match.group(1)

def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G


def run_experiment():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    solver = SimulatedAnnealingSampler()

    all_rows = []

    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".pkl") or filename == "quantum_extra.pkl":
            continue

        with open(os.path.join(DATASET_PATH, filename), "rb") as f:
            data = pickle.load(f)

        total_graphs = len(data["graphs"])
        density = extract_density(filename)
        config = TUNED_CONFIGS[density]
        print(f"[{filename}] N={data['num_vertices']} | {total_graphs} graphs")

        for graph_id, graph in enumerate(data["graphs"]):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            lower_bound = graph["lower_bound"]

            # Evaluate both penalty methods for every graph
            for penalty in PENALTY_METHODS:
                bqm = minla.generate_bqm_instance(G, penalty_mode=penalty)

                feasible_costs = []
                total_elapsed = 0

                for seed in SEEDS:
                    t0 = time.time()
                    sampleset = solver.sample(
                        bqm,
                        num_reads=NUM_READS,
                        num_sweeps=NUM_SWEEPS,
                        seed=seed,
                        beta_schedule_type=config["beta_schedule_type"],
                        beta_range=config["beta_range"]
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
                total_reads = len(SEEDS) * NUM_READS
                avg_minla_cost = np.mean(feasible_costs) if feasible else None
                approx_ratio = avg_minla_cost / lower_bound if feasible else None

                all_rows.append({
                    "n": n,
                    "m": m,
                    "graph_id": graph_id,
                    "solver": "SimulatedAnnealingSampler",
                    "penalty": penalty,
                    "feasible": feasible,
                    "feasible_read_count": feasible_read_count,
                    "avg_minla_cost": avg_minla_cost,
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

                print(
                    f"  [{filename}] graph {graph_id + 1}/{total_graphs} | penalty={penalty.upper():5} | "
                    f"feasible={feasible!s:5} | reads={feasible_read_count}/{total_reads} | "
                    f"time={total_elapsed:.2f}s"
                )

    df = pd.DataFrame(all_rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_PATH, f"sa_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary -> {csv_path}")


if __name__ == "__main__":
    run_experiment()