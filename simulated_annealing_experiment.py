import os
import pickle
import time
import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler, PathIntegralAnnealingSampler
import Utils.MinLA as minla
import networkx as nx

DATASET_PATH = "Dataset/quantum_dataset"
RESULTS_PATH = "Results/sa_experiment"

SEEDS = [42, 123, 456, 789, 999]
NUM_READS = 10
NUM_SWEEPS = 1000


def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G


def run_experiment():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    solver = SimulatedAnnealingSampler()
    # solver = PathIntegralAnnealingSampler()

    all_rows = []

    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".pkl"):
            continue

        with open(os.path.join(DATASET_PATH, filename), "rb") as f:
            data = pickle.load(f)

        total_graphs = len(data["graphs"])
        print(f"[{filename}] N={data['num_vertices']} | {total_graphs} graphs")

        for graph_id, graph in enumerate(data["graphs"]):
            G = convert_graph_data_to_nx(graph)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            lower_bound = graph["lower_bound"]
            bqm = minla.generate_bqm_instance(G)

            best_feasible_costs = []
            feasible_seed_count = 0
            total_elapsed = 0

            for seed in SEEDS:
                t0 = time.time()
                # sampleset = solver.sample(bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, seed=seed, beta_schedule_type="custom", Hp_field=Hp_field, Hd_field=Hd_field)                
                sampleset = solver.sample(bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, seed=seed, beta_schedule_type="linear", beta_range=(1e-1, 5e-1))
                elapsed = time.time() - t0
                total_elapsed += elapsed

                best_cost = None
                for sample in sampleset.samples():
                    ordering, is_feasible = minla.decode_solution(sample, n)
                    if is_feasible:
                        cost = minla.calculate_min_linear_arrangement(G, ordering)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost

                if best_cost is not None:
                    best_feasible_costs.append(best_cost)
                    feasible_seed_count += 1

            feasible = len(best_feasible_costs) > 0
            avg_minla_cost = np.mean(best_feasible_costs) if feasible else None
            approx_ratio = avg_minla_cost / lower_bound if feasible else None

            all_rows.append({
                "n": n,
                "m": m,
                "graph_id": graph_id,
                "solver": "SimulatedAnnealingSampler",
                "feasible": feasible,
                "feasible_seed_count": feasible_seed_count,
                "avg_minla_cost": avg_minla_cost,
                "lower_bound": lower_bound,
                "approx_ratio": approx_ratio,
                "time_s": round(total_elapsed, 3),
                "num_seeds": len(SEEDS),
                "num_reads": NUM_READS,
                "num_sweeps": NUM_SWEEPS,
            })

            print(
                f"  [{filename}] graph {graph_id + 1}/{total_graphs} | "
                f"feasible={feasible} | seeds={feasible_seed_count}/{len(SEEDS)} | "
                f"time={total_elapsed:.2f}s"
            )

    df = pd.DataFrame(all_rows)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_PATH, f"sa_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV summary -> {csv_path}")


if __name__ == "__main__":
    run_experiment()
