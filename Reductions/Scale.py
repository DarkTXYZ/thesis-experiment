import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import Utils.MinLA as minla
import numpy as np

from dwave.samplers import PathIntegralAnnealingSampler

if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(30, 0.5, seed=40)
    n = graph.number_of_nodes()

    bqm = minla.generate_bqm_instance(graph)
    
    bqm.scale(0.1)

    solver = PathIntegralAnnealingSampler()
    sampleset = solver.sample(bqm, seed = 42, num_reads=10, num_sweeps=1000, beta_schedule_type='linear')
    
    best_solution = dict(sampleset.first.sample)
    best_energy = sampleset.first.energy
    
    print(f"Best energy after reduction: {best_energy}")
    # print(f"Best solution after reduction: {best_solution}")
    
    # print the solution matrix
    # for u in range(n):
    #     row = []
    #     for k in range(n):
    #         row.append(best_solution.get(f'X[{u}][{k}]', 0))
    #     print(f"u={u}: {row}")
        
    # print ordering of vertices based on the solution
    ordering = []
    for u in range(n):
        ordering.append(int(sum(best_solution.get(f'X[{u}][{k}]', 0) for k in range(n))))
    print(f"Ordering of vertices based on solution: {ordering}")
    
    # Check feasibility (like in DR.py)
    sol_matrix = np.array([
        [best_solution.get(f'X[{u}][{k}]', 0) for k in range(n)]
        for u in range(n)
    ], dtype=int)
    
    # Check thermometer constraint: X[u][0] >= X[u][1] >= ... >= X[u][n-1]
    # (no 0 followed by 1 in any row)
    is_feasible = True
    for u in range(n):
        if np.any((sol_matrix[u, :-1] == 0) & (sol_matrix[u, 1:] == 1)):
            is_feasible = False
            break
    
    # Check bijection constraint: each vertex has unique position from 1 to n
    if is_feasible:
        labels = set(np.sum(sol_matrix, axis=1))
        is_feasible = labels == set(range(1, n + 1))
    
    feasible_label = "✓ FEASIBLE" if is_feasible else "✗ INFEASIBLE"
    print(f"Solution feasibility: {feasible_label}")
    
    print(minla.calculate_min_linear_arrangement(graph, ordering))
    
    