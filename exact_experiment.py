import csv
import os
import pickle

import dimod
import networkx as nx
import numpy as np
from dwave.samplers import PathIntegralAnnealingSampler

from Utils.MinLA import generate_bqm_instance, calculate_min_linear_arrangement, decode_solution

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
PIA_SEED = 42
Hp_field = np.linspace(0, 1, PIA_NUM_SWEEPS)
Hd_field = np.linspace(1, 0, PIA_NUM_SWEEPS)

def solve_with_exact(graph, bqm : dimod.BinaryQuadraticModel, n):
    sampler = dimod.ExactSolver()
    response = sampler.sample(bqm)

    feasible_solutions = []
    infeasible_energies = []  # Track energies of invalid assignments

    for sample, energy in zip(response.samples(), bqm.energies(response.samples())):
        ordering, is_feasible = decode_solution(dict(sample), n)
        if is_feasible:
            cost = calculate_min_linear_arrangement(graph, ordering)
            
            # Store the tuple of (ordering, MinLA cost, BQM energy)
            feasible_solutions.append((tuple(ordering), cost, energy))
        else:
            infeasible_energies.append(energy)

    min_infeasible_energy = min(infeasible_energies) if infeasible_energies else None

    return feasible_solutions, min_infeasible_energy


def solve_with_pia(graph, bqm, n):
    sampler = PathIntegralAnnealingSampler()

    response = sampler.sample(bqm, num_reads=PIA_NUM_READS, num_sweeps=PIA_NUM_SWEEPS, beta_schedule_type="custom", Hp_field=Hp_field, Hd_field=Hd_field, seed=PIA_SEED)

    best_cost = None
    feasible_count = 0
    for sample, energy in zip(response.samples(), response.data_vectors["energy"]):
        ordering, is_feasible = decode_solution(dict(sample), n)
        if is_feasible:
            feasible_count += 1
            cost = calculate_min_linear_arrangement(graph, ordering)
            if best_cost is None or cost < best_cost:
                best_cost = cost

    return best_cost, feasible_count


def report(solver_name, graph_name, graph, gt_obj, gt_solutions, feasible_solutions, min_infeasible_energy, penalty_mode):
    if not feasible_solutions:
        print(f"  {solver_name}: No feasible solutions found!")
        if min_infeasible_energy is not None:
            print(f"    Lowest invalid energy: {min_infeasible_energy}")
        return

    best_cost = min(s[1] for s in feasible_solutions)
    best_feasible_energy = min(s[2] for s in feasible_solutions)
    optimal_solutions = [s[0] for s in feasible_solutions if s[1] == best_cost]

    gt_set = set(gt_solutions)
    found_set = set(optimal_solutions)

    is_optimal = best_cost == gt_obj
    match = gt_set == found_set

    print(f"  {solver_name} (penalty: {penalty_mode}):")
    print(f"    Best obj: {best_cost} (optimal: {is_optimal})")
    print(f"    Best feasible BQM energy: {best_feasible_energy}")
    
    # Check if an invalid state beats the valid state
    if min_infeasible_energy is not None:
        print(f"    Lowest invalid BQM energy: {min_infeasible_energy}")
        if min_infeasible_energy < best_feasible_energy:
            print(f"    🚨 WARNING: Invalid states exist with LOWER energy than the best valid state!")
        elif min_infeasible_energy == best_feasible_energy:
            print(f"    ⚠️ WARNING: Invalid states exist with EQUAL energy to the best valid state!")

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

        print(f"\n{'='*50}")
        print(f"Graph: {graph_name} (n={n}, m={graph.number_of_edges()}), ground truth obj: {gt_obj}")

        for penalty_mode in ['lucas', 'exact']:
            bqm = generate_bqm_instance(graph, penalty_mode)

            # dimod ExactSolver
            exact_results, min_infeasible_energy = solve_with_exact(graph, bqm, n)
            report("dimod.ExactSolver", graph_name, graph, gt_obj, gt_solutions, exact_results, min_infeasible_energy, penalty_mode)

            if exact_results:
                exact_best = min(s[1] for s in exact_results)
                exact_optimal = set(s[0] for s in exact_results if s[1] == exact_best)
                exact_match = gt_set == exact_optimal
            else:
                exact_best = None
                exact_optimal = set()
                exact_match = False

            # Path Integral Annealing
            pia_best, pia_feasible_count = solve_with_pia(graph, bqm, n)
            if pia_best is not None:
                print(f"  PathIntegralAnnealing (penalty: {penalty_mode}): Best obj: {pia_best} (optimal: {pia_best == gt_obj})")
                print(f"    Total feasible: {pia_feasible_count}")
            else:
                print(f"  PathIntegralAnnealing (penalty: {penalty_mode}): No feasible solutions found!")

            m = graph.number_of_edges()
            density = round(2 * m / (n * (n - 1)), 4) if n > 1 else 0
            display_name = GRAPH_DISPLAY_NAMES.get(graph_name, graph_name)

            rows.append({
                "Graph": display_name,
                "Penalty Mode": penalty_mode,
                "|V|": n,
                "|E|": m,
                "Density": density,
                "QUBO Variables": n * n,
                "Optimal MinLA": gt_obj,
                "# Optimal Permutations": len(gt_set),
                "ExactSolver MinLA": exact_best,
                "ExactSolver # Feasible": len(exact_results),
                "ExactSolver # Optimal Found": len(exact_optimal),
                "ExactSolver All Found": exact_match,
                "PIA MinLA": pia_best,
                "PIA # Feasible": pia_feasible_count,
                "PIA Optimal": pia_best == gt_obj if pia_best is not None else False,
            })

    fieldnames = [
        "Graph", "Penalty Mode", "|V|", "|E|", "Density", "QUBO Variables",
        "Optimal MinLA", "# Optimal Permutations",
        "ExactSolver MinLA", "ExactSolver # Feasible", "ExactSolver # Optimal Found", "ExactSolver All Found",
        "PIA MinLA", "PIA # Feasible", "PIA Optimal",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
