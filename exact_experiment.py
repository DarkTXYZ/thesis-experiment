import csv
import math
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

BQM_DISPLAY_NAMES = {
    "exact": "ExactBQM",
    "lucas": "LucasBQM",
}

PIA_NUM_READS = 25
PIA_NUM_SWEEPS = 500
PIA_SEED = 42
Hp_field = np.linspace(0, 1, PIA_NUM_SWEEPS)
Hd_field = np.linspace(1, 0, PIA_NUM_SWEEPS)

def solve_with_exact(graph, bqm : dimod.BinaryQuadraticModel, n):
    """Enumerate every binary assignment of the BQM with dimod.ExactSolver.

    Returns:
        feasible_solutions: list of (ordering, MinLA cost, BQM energy) for
            every assignment that decodes to a valid linear arrangement.
        ground_state_energy: the minimum BQM energy over ALL assignments
            (feasible and infeasible).
        ground_state_count: number of assignments (feasible or not) attaining
            ground_state_energy.
        infeasible_ground_state: True if an infeasible assignment attains
            ground_state_energy (would indicate the formulation is broken).
    """
    sampler = dimod.ExactSolver()
    response = sampler.sample(bqm)

    feasible_solutions = []
    all_energies = []

    for sample, energy in zip(response.samples(), bqm.energies(response.samples())):
        ordering, is_feasible = decode_solution(dict(sample), n)
        all_energies.append((energy, is_feasible))
        if is_feasible:
            # decode_solution returns 1-indexed positions (1..n); shift to
            # 0-indexed (0..n-1) to match CpSolver's position convention.
            ordering = ordering - 1
            cost = calculate_min_linear_arrangement(graph, ordering)

            # Store the tuple of (ordering, MinLA cost, BQM energy)
            feasible_solutions.append((tuple(ordering), cost, energy))

    ground_state_energy = min(e for e, _ in all_energies)
    ground_state_count = sum(1 for e, _ in all_energies if e == ground_state_energy)
    infeasible_ground_state = any(
        e == ground_state_energy and not is_feasible for e, is_feasible in all_energies
    )

    return feasible_solutions, ground_state_energy, ground_state_count, infeasible_ground_state


def solve_with_pia(graph, bqm, n):
    sampler = PathIntegralAnnealingSampler()

    response = sampler.sample(bqm, num_reads=PIA_NUM_READS, num_sweeps=PIA_NUM_SWEEPS, beta_schedule_type="custom", Hp_field=Hp_field, Hd_field=Hd_field, seed=PIA_SEED)

    best_cost = None
    feasible_count = 0
    for sample, energy in zip(response.samples(), response.data_vectors["energy"]):
        ordering, is_feasible = decode_solution(dict(sample), n)
        if is_feasible:
            feasible_count += 1
            ordering = ordering - 1
            cost = calculate_min_linear_arrangement(graph, ordering)
            if best_cost is None or cost < best_cost:
                best_cost = cost

    return best_cost, feasible_count


def report(solver_name, graph_name, graph, gt_obj, gt_solutions, feasible_solutions,
           ground_state_energy, ground_state_count, infeasible_ground_state, n, bqm_name):
    expected_feasible = math.factorial(n)

    print(f"  {solver_name} ({bqm_name}):")

    if not feasible_solutions:
        print(f"    No feasible solutions found!")
        print(f"    Ground-state energy: {ground_state_energy} (attained by an infeasible assignment)")
        return

    best_cost = min(s[1] for s in feasible_solutions)
    optimal_solutions = [s[0] for s in feasible_solutions if s[1] == best_cost]

    gt_set = set(gt_solutions)
    found_set = set(optimal_solutions)

    is_optimal = best_cost == gt_obj
    match = gt_set == found_set
    approx_ratio = best_cost / gt_obj if gt_obj else None

    print(f"    Total feasible assignments: {len(feasible_solutions)} "
          f"(expected {n}! = {expected_feasible}, matches: {len(feasible_solutions) == expected_feasible})")
    print(f"    Ground-state energy: {ground_state_energy} (# ground states: {ground_state_count})")
    if infeasible_ground_state:
        print(f"    🚨 WARNING: an infeasible assignment attains the ground-state energy!")
    print(f"    Best feasible MinLA cost: {best_cost} (optimal: {is_optimal}, approx. ratio: {approx_ratio})")
    print(f"    #optimal solutions found: {len(found_set)} / {len(gt_set)}")
    print(f"    Exact match with CpSolver ground truth: {match}")
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

        for penalty_mode in ['exact', 'lucas']:
            bqm_name = BQM_DISPLAY_NAMES[penalty_mode]
            bqm = generate_bqm_instance(graph, penalty_mode)

            # dimod ExactSolver: exhaustive enumeration of every binary assignment
            exact_results, gs_energy, gs_count, infeasible_gs = solve_with_exact(graph, bqm, n)
            report("dimod.ExactSolver", graph_name, graph, gt_obj, gt_solutions, exact_results,
                   gs_energy, gs_count, infeasible_gs, n, bqm_name)

            if exact_results:
                exact_best = min(s[1] for s in exact_results)
                exact_optimal = set(s[0] for s in exact_results if s[1] == exact_best)
                exact_match = gt_set == exact_optimal
                exact_ratio = exact_best / gt_obj if gt_obj else None
            else:
                exact_best = None
                exact_optimal = set()
                exact_match = False
                exact_ratio = None

            # Path Integral Annealing: heuristic sanity check
            pia_best, pia_feasible_count = solve_with_pia(graph, bqm, n)
            pia_ratio = pia_best / gt_obj if (pia_best is not None and gt_obj) else None
            if pia_best is not None:
                print(f"  PathIntegralAnnealing ({bqm_name}): Best obj: {pia_best} "
                      f"(optimal: {pia_best == gt_obj}, approx. ratio: {pia_ratio})")
                print(f"    Feasible reads: {pia_feasible_count} / {PIA_NUM_READS}")
            else:
                print(f"  PathIntegralAnnealing ({bqm_name}): No feasible solutions found!")

            m = graph.number_of_edges()
            density = round(2 * m / (n * (n - 1)), 4) if n > 1 else 0
            display_name = GRAPH_DISPLAY_NAMES.get(graph_name, graph_name)

            rows.append({
                "Graph": display_name,
                "BQM Type": bqm_name,
                "|V|": n,
                "|E|": m,
                "Density": density,
                "QUBO Variables": n * n,
                "Optimal MinLA (CpSolver)": gt_obj,
                "# Optimal Permutations": len(gt_set),
                "N!": math.factorial(n),
                "ExactSolver # Feasible": len(exact_results),
                "ExactSolver Feasible == N!": len(exact_results) == math.factorial(n),
                "ExactSolver Ground-State Energy": gs_energy,
                "ExactSolver # Ground States": gs_count,
                "ExactSolver Infeasible Ground State": infeasible_gs,
                "ExactSolver MinLA": exact_best,
                "ExactSolver Approx Ratio": exact_ratio,
                "ExactSolver # Optimal Found": len(exact_optimal),
                "ExactSolver Exact Match": exact_match,
                "PIA MinLA": pia_best,
                "PIA # Feasible Reads": pia_feasible_count,
                "PIA Approx Ratio": pia_ratio,
            })

    fieldnames = [
        "Graph", "BQM Type", "|V|", "|E|", "Density", "QUBO Variables",
        "Optimal MinLA (CpSolver)", "# Optimal Permutations", "N!",
        "ExactSolver # Feasible", "ExactSolver Feasible == N!",
        "ExactSolver Ground-State Energy", "ExactSolver # Ground States", "ExactSolver Infeasible Ground State",
        "ExactSolver MinLA", "ExactSolver Approx Ratio", "ExactSolver # Optimal Found", "ExactSolver Exact Match",
        "PIA MinLA", "PIA # Feasible Reads", "PIA Approx Ratio",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
