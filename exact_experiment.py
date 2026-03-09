import csv
import os
import pickle

import dimod
import networkx as nx
import numpy as np
from dwave.samplers import PathIntegralAnnealingSampler

from Utils.MinLA import generate_bqm_instance, calculate_min_linear_arrangement

DATASET_DIR = "Dataset/exact_dataset"
PKL_PATH = os.path.join(DATASET_DIR, "exact_dataset.pkl")
CSV_PATH = os.path.join("Results", "exact_experiment_results.csv")

GRAPH_DISPLAY_NAMES = {
    "c3": "C_3", "c4": "C_4",
    "k4": "K_4",
    "p3": "P_3", "p4": "P_4",
    "s4": "S_4",
    "g01": "G_01", "g02": "G_02",
}

PIA_NUM_READS = 10
PIA_NUM_SWEEPS = 1000


def decode_thermometer(sample, n):
    """Decode thermometer-encoded sample to node ordering (0-indexed)."""
    sol = np.array([
        [sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
        for u in range(n)
    ], dtype=int)

    # Check thermometer validity: no 0 followed by 1
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return None, False

    labels = np.sum(sol, axis=1)  # 1..n in thermometer
    if set(labels) != set(range(1, n + 1)):
        return None, False

    ordering = labels - 1  # convert to 0-indexed
    return ordering.tolist(), True


def solve_with_exact(graph, bqm, n):
    sampler = dimod.ExactSolver()
    response = sampler.sample(bqm)

    feasible_solutions = []
    for sample, energy in zip(response.samples(), response.data_vectors["energy"]):
        ordering, is_feasible = decode_thermometer(dict(sample), n)
        if is_feasible:
            cost = calculate_min_linear_arrangement(graph, ordering)
            feasible_solutions.append((tuple(ordering), cost, energy))

    return feasible_solutions


def solve_with_pia(graph, bqm, n):
    sampler = PathIntegralAnnealingSampler()
    response = sampler.sample(bqm, num_reads=PIA_NUM_READS, num_sweeps=PIA_NUM_SWEEPS)

    best_cost = None
    for sample, energy in zip(response.samples(), response.data_vectors["energy"]):
        ordering, is_feasible = decode_thermometer(dict(sample), n)
        if is_feasible:
            cost = calculate_min_linear_arrangement(graph, ordering)
            if best_cost is None or cost < best_cost:
                best_cost = cost

    return best_cost


def report(solver_name, graph_name, graph, gt_obj, gt_solutions, feasible_solutions):
    if not feasible_solutions:
        print(f"  {solver_name}: No feasible solutions found!")
        return

    best_cost = min(s[1] for s in feasible_solutions)
    optimal_solutions = [s[0] for s in feasible_solutions if s[1] == best_cost]

    gt_set = set(gt_solutions)
    found_set = set(optimal_solutions)

    is_optimal = best_cost == gt_obj
    match = gt_set == found_set

    print(f"  {solver_name}:")
    print(f"    Best obj: {best_cost} (optimal: {is_optimal})")
    print(f"    #optimal solutions found: {len(found_set)} / {len(gt_set)}")
    print(f"    Total feasible: {len(feasible_solutions)}")
    print(f"    Exact match: {match}")
    if not match:
        missing = gt_set - found_set
        extra = found_set - gt_set
        if missing:
            print(f"    Missing: {len(missing)} solutions")
        if extra:
            print(f"    Extra: {len(extra)} solutions")


def main():
    with open(PKL_PATH, "rb") as f:
        ground_truth = pickle.load(f)

    edgelist_files = sorted(
        f for f in os.listdir(DATASET_DIR) if f.endswith(".edgelist")
    )

    rows = []

    for filename in edgelist_files:
        graph_name = os.path.splitext(filename)[0]
        filepath = os.path.join(DATASET_DIR, filename)
        graph = nx.read_edgelist(filepath, nodetype=int)
        n = graph.number_of_nodes()

        gt = ground_truth[graph_name]
        gt_obj = gt["objective_value"]
        gt_solutions = [tuple(s) for s in gt["solutions"]]
        gt_set = set(gt_solutions)

        bqm = generate_bqm_instance(graph)

        print(f"\n{'='*50}")
        print(f"Graph: {graph_name} (n={n}, m={graph.number_of_edges()}), ground truth obj: {gt_obj}")

        # dimod ExactSolver
        exact_results = solve_with_exact(graph, bqm, n)
        report("dimod.ExactSolver", graph_name, graph, gt_obj, gt_solutions, exact_results)

        if exact_results:
            exact_best = min(s[1] for s in exact_results)
            exact_optimal = set(s[0] for s in exact_results if s[1] == exact_best)
            exact_match = gt_set == exact_optimal
        else:
            exact_best = None
            exact_optimal = set()
            exact_match = False

        # Path Integral Annealing
        pia_best = solve_with_pia(graph, bqm, n)
        if pia_best is not None:
            print(f"  PathIntegralAnnealing: Best obj: {pia_best} (optimal: {pia_best == gt_obj})")
        else:
            print(f"  PathIntegralAnnealing: No feasible solutions found!")

        m = graph.number_of_edges()
        density = round(2 * m / (n * (n - 1)), 4) if n > 1 else 0
        display_name = GRAPH_DISPLAY_NAMES.get(graph_name, graph_name)

        rows.append({
            "Graph": display_name,
            "|V|": n,
            "|E|": m,
            "Density": density,
            "QUBO Variables": n * n,
            "Optimal MinLA": gt_obj,
            "# Optimal Permutations": len(gt_set),
            "ExactSolver MinLA": exact_best,
            "ExactSolver # Optimal Found": len(exact_optimal),
            "ExactSolver All Found": exact_match,
            "PIA MinLA": pia_best,
            "PIA Optimal": pia_best == gt_obj if pia_best is not None else False,
        })

    fieldnames = [
        "Graph", "|V|", "|E|", "Density", "QUBO Variables",
        "Optimal MinLA", "# Optimal Permutations",
        "ExactSolver MinLA", "ExactSolver # Optimal Found", "ExactSolver All Found",
        "PIA MinLA", "PIA Optimal",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
