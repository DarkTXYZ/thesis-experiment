import os
import pickle
import re
import time
import pandas as pd
from dwave.samplers import RandomSampler
import Utils.MinLA as minla
import networkx as nx

DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_PATH = "Results/random_experiment"

SEEDS = [42]
NUM_READS = 25
PENALTY_METHODS = ["lucas", "exact"]

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

def run_experiment():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Initialize the uninformed random sampler
    solver = RandomSampler()

    all_rows = []

    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".pkl") or filename == "quantum_extra.pkl":
            continue

        with open(os.path.join(DATASET_PATH, filename), "rb") as f:
            data = pickle.load(f)

        total_graphs = len(data["graphs"])
        density = extract_density(filename)
        print(f"[{filename}] N={data['num_vertices']} | {total_graphs} graphs")

        for graph_id, graph in enumerate(data["graphs"]):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            lower_bound = graph["lower_bound"]

            # Evaluate both penalty methods, identical to SA and PIA scripts
            for penalty in PENALTY_METHODS:
                bqm = minla.generate_bqm_instance(G, penalty_mode=penalty)

                feasible_costs = []
                total_elapsed = 0

                for seed in SEEDS:
                    t0 = time.time()
                    # RandomSampler only requires the BQM and the number of reads.
                    # We pass the seed for strict computational reproducibility.
                    sampleset = solver.sample(
                        bqm,
                        num_reads=NUM_READS,
                        seed=seed
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
                    "solver": "RandomSampler",
                    "penalty": penalty,
                    "feasible": feasible,
                    "feasible_read_count": feasible_read_count,
                    "best_minla_cost": best_minla_cost,
                    "lower_bound": lower_bound,
                    "approx_ratio": approx_ratio,
                    "time_s": round(total_elapsed, 3),
                    "num_seeds": len(SEEDS),
                    "num_reads": NUM_READS,
                    # Parameter-free baseline, so hyperparameter columns are omitted
                })

                print(
                    f"  [{filename}] graph {graph_id + 1}/{total_graphs} | penalty={penalty.upper():5} | "
                    f"feasible={feasible!s:5} | reads={feasible_read_count}/{total_reads} | "
                    f"time={total_elapsed:.2f}s"
                )

    df = pd.DataFrame(all_rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_PATH, f"random_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary -> {csv_path}")

if __name__ == "__main__":
    run_experiment()