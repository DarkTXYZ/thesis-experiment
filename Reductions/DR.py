import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Utils.MinLA as minla
import networkx as nx
import dimod
import numpy as np
from dwave.samplers import PathIntegralAnnealingSampler
from dwave.embedding.chain_strength import uniform_torque_compensation
import json

def DR(QUBO):
    HAV = -float('inf')
    LAV = float('inf')
    
    for var, val in QUBO.linear.items():
        if val == 0:
            continue
        if abs(val) < LAV:
            LAV = abs(val)
        if abs(val) > HAV:
            HAV = abs(val)
            
    for vars, val in QUBO.quadratic.items():
        if val == 0:
            continue
        if abs(val) < LAV:
            LAV = abs(val)
        if abs(val) > HAV:
            HAV = abs(val)
            
    return HAV / LAV, 2 * LAV


def split_QUBO(QUBO, TDR):    
    QUBO_sub_1 = dimod.BinaryQuadraticModel('BINARY')
    QUBO_sub_2 = dimod.BinaryQuadraticModel('BINARY')
    
    for var, val in QUBO.linear.items():
        if abs(val) > TDR:
            QUBO_sub_1.add_linear(var, val / 2)
            QUBO_sub_2.add_linear(var, val / 2)
        else:
            QUBO_sub_1.add_linear(var, val)        
            
    for vars, val in QUBO.quadratic.items():
        if abs(val) > TDR:
            QUBO_sub_1.add_quadratic(vars[0], vars[1], val / 2)
            QUBO_sub_2.add_quadratic(vars[0], vars[1], val / 2)
        else:
            QUBO_sub_1.add_quadratic(vars[0], vars[1], val)
            
    return QUBO_sub_1, QUBO_sub_2


def decode_minla_solution(solution_dict, n):
    """
    Decode MinLA solution from BQM variables (thermometer encoding).
    X[u][k] = 1 means vertex u has position <= k.
    
    Args:
        solution_dict: Dictionary with keys like 'X[u][k]' and binary values
        n: Number of vertices
        
    Returns:
        tuple: (ordering, feasible)
               ordering: array where ordering[u] is the position (label) of vertex u (1-indexed)
               feasible: boolean indicating if solution satisfies thermometer and bijection constraints
    """
    # Build 2D solution matrix sol[u][k] = X[u][k]
    sol = np.array([
        [solution_dict.get(f'X[{u}][{k}]', 0) for k in range(n)]
        for u in range(n)
    ], dtype=int)
    
    # Check feasibility: thermometer constraint
    # X[u][0] >= X[u][1] >= ... >= X[u][n-1] (no 0 followed by 1 in any row)
    is_feasible = True
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            is_feasible = False
            break
    
    # Check bijection constraint: each vertex has a unique position from 1 to n
    if is_feasible:
        labels = set(np.sum(sol, axis=1))
        is_feasible = labels == set(range(1, n + 1))
    
    # Ordering: position label of each vertex (sum of thermometer values)
    ordering = np.sum(sol, axis=1)
    
    return ordering, is_feasible


def solve_and_display_solutions(Q_sol, graph, n, mode_name=""):
    """
    Solve all BQMs and display feasible solutions with lowest energy.
    
    Args:
        Q_sol: List of BQMs to solve
        graph: NetworkX graph
        n: Number of vertices
        mode_name: Name of the solving mode for display purposes
    """
    # solve all bqm in Q_sol with PathIntegral
    solver = PathIntegralAnnealingSampler()
    results = []
    
    for idx, bqm in enumerate(Q_sol):
        chain_strength = uniform_torque_compensation(bqm)
        sampleset = solver.sample(
            bqm=bqm,
            num_reads=10,
            num_sweeps=1000,
            seed=42,
            beta_schedule_type='linear'
        )
        results.append(sampleset)
        print(f"Solved BQM {idx + 1}/{len(Q_sol)}: Best energy = {sampleset.first.energy}")
    
    print(f"\nTotal BQMs solved: {len(results)}")
    if mode_name:
        print(f"Mode: {mode_name}")
    
    # Find global lowest energy across all BQMs
    lowest_energy = float('inf')
    for idx, sampleset in enumerate(results):
        min_energy = sampleset.first.energy
        if min_energy < lowest_energy:
            lowest_energy = min_energy
    
    print(f"\nGlobal lowest energy: {lowest_energy}")
    
    # For each BQM with lowest energy, find all solutions
    best_solutions = []  # List of (bqm_idx, solution_dict, energy, feasible) tuples
    
    for idx, sampleset in enumerate(results):
        # Check if this BQM has solutions with global lowest energy
        bqm_lowest_energy = sampleset.first.energy
        
        if bqm_lowest_energy == lowest_energy:
            print(f"\n  BQM {idx} achieved lowest energy:")
            
            # Find all solutions with lowest energy in this BQM
            all_solutions_in_bqm = []
            
            for sample in sampleset.data(['sample', 'energy']):
                if sample.energy == bqm_lowest_energy:
                    solution_dict = dict(sample.sample)
                    ordering, feasible = decode_minla_solution(solution_dict, n)
                    
                    all_solutions_in_bqm.append((solution_dict, ordering, sample.energy, feasible))
            
            print(f"    Found {len(all_solutions_in_bqm)} solutions with energy {bqm_lowest_energy}")
            
            for sol_dict, ordering, energy, feasible in all_solutions_in_bqm:
                best_solutions.append((idx, sol_dict, energy, feasible))
    
    print(f"\n{'='*70}")
    print(f"All solutions with lowest energy = {lowest_energy}:")
    print(f"Total solutions found: {len(best_solutions)}")
    print(f"{'='*70}")
    
    # Decode and display each solution
    for sol_idx, (bqm_idx, best_solution, energy, feasible) in enumerate(best_solutions):
        feasible_label = "✓ FEASIBLE" if feasible else "✗ INFEASIBLE"
        print(f"\n{'='*70}")
        print(f"Lowest Energy Solution {sol_idx + 1}/{len(best_solutions)} [{feasible_label}]")
        print(f"{'='*70}")
        print(f"BQM index: {bqm_idx}")
        print(f"Energy: {energy}")
        
        # Decode solution in MinLA labeling
        ordering, is_feasible = decode_minla_solution(best_solution, n)
        
        print(f"Feasible: {is_feasible}")
        print(f"Ordering (position of each vertex): {list(ordering)}")
        
        # Display solution in matrix form
        print(f"\nSolution Matrix (X[u][k]) - Thermometer Encoding:")
        print(f"Rows = vertices (u), Columns = positions (k)")
        
        sol_matrix = np.array([
            [best_solution.get(f'X[{u}][{k}]', 0) for k in range(n)]
            for u in range(n)
        ], dtype=int)
        
        # Print column headers
        print("\n     ", end="")
        for k in range(n):
            print(f"k={k:2d} ", end="")
        print()
        
        # Print matrix rows
        for u in range(n):
            print(f"u={u:2d} ", end="")
            for k in range(n):
                print(f" {sol_matrix[u][k]:2d}  ", end="")
            print(f"  pos={ordering[u]}")
        
        # Calculate MinLA cost
        if all(v is not None for v in ordering):
            minla_cost = minla.calculate_min_linear_arrangement(graph, list(ordering))
            print(f"\nMinLA cost (edge sum of distances): {minla_cost}")
        else:
            print(f"\nWarning: Incomplete assignment in solution")


def mode_direct_solve(graph):
    """Mode 1: Solve QUBO directly without splitting."""
    print("="*70)
    print("MODE 1: Direct Solve (No Splitting)")
    print("="*70)
    
    n = graph.number_of_nodes()
    Q = minla.generate_bqm_instance(graph)
    
    print(f"Original QUBO generated with {n} vertices\n")
    
    Q_sol = [Q]  # Directly use the original QUBO
    solve_and_display_solutions(Q_sol, graph, n, mode_name="Direct Solve")


def mode_split_solve(graph):
    """Mode 2: Split QUBO and solve each sub-QUBO."""
    print("="*70)
    print("MODE 2: Split & Solve")
    print("="*70)
    
    n = graph.number_of_nodes()
    Q = minla.generate_bqm_instance(graph)
    
    _, TDR = DR(Q)
    print(f"Dynamic Range Ratio (DR) threshold: {TDR}")
    print(f"Original QUBO dynamic range exceeds threshold - will split recursively\n")
    
    Q_ope = [Q]
    Q_sol = []
    split_count = 0
    
    # Split QUBO recursively
    while len(Q_ope) > 0:
        q_ope = Q_ope.pop(0)
        dr, _ = DR(q_ope)
        if dr <= TDR:
            Q_sol.append(q_ope)
        else:
            split_count += 1
            print(f"Splitting QUBO {split_count} (DR={dr:.4f} > TDR={TDR:.4f})...")
            q_ope1, q_ope2 = split_QUBO(q_ope, TDR)
            Q_ope.append(q_ope1)
            Q_ope.append(q_ope2)
    
    print(f"\n{'='*70}")
    print(f"Splitting complete!")
    print(f"Original QUBO split into {len(Q_sol)} sub-QUBOs")
    print(f"{'='*70}\n")
    
    solve_and_display_solutions(Q_sol, graph, n, mode_name="Split Solve")

if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(30, 0.5, seed=40)
    
    # ========== MODE SELECTOR ==========
    # Change this to 1 or 2 to select the mode:
    # 1 = Direct solve (no splitting)
    # 2 = Split solve (recursive splitting)
    MODE = 1
    # ===================================
    
    if MODE == 1:
        mode_direct_solve(graph)
    elif MODE == 2:
        mode_split_solve(graph)
    else:
        print(f"Invalid MODE: {MODE}. Choose 1 or 2.")




