import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import Utils.MinLA as minla

from dwave.samplers import PathIntegralAnnealingSampler

if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(10, 0.5, seed=40)
    n = graph.number_of_nodes()

    bqm = minla.generate_bqm_instance(graph)

    print(len(bqm.linear) + len(bqm.quadratic))
    
    reduction_rate = 0
    number_of_coefficients = len(bqm.linear) + len(bqm.quadratic)
    
    # set the lowest (reduction_rate * number_of_coefficients) linear and quadratic coefficients to zero
    all_coefficients = []
    
    for var, coeff in bqm.linear.items():
        all_coefficients.append((coeff, var, 'linear'))

    for (var1, var2), coeff in bqm.quadratic.items():
        all_coefficients.append((coeff, (var1, var2), 'quadratic'))
        
    all_coefficients.sort(key=lambda x: abs(x[0]), reverse=True)
    
    num_to_zero = int(reduction_rate * number_of_coefficients)
    for i in range(num_to_zero):
        coeff, var, coeff_type = all_coefficients[i]
        if coeff_type == 'linear':
            bqm.set_linear(var, 0.0)
        else:
            bqm.set_quadratic(var[0], var[1], 0.0)
            
    solver = PathIntegralAnnealingSampler()
    sampleset = solver.sample(bqm, seed = 42, num_reads=10, num_sweeps=1000, beta_schedule_type='linear')
    
    best_solution = dict(sampleset.first.sample)
    best_energy = sampleset.first.energy
    
    print(f"Best energy after reduction: {best_energy}")
    print(f"Best solution after reduction: {best_solution}")
    
    # print the solution matrix
    for u in range(n):
        row = []
        for k in range(n):
            row.append(best_solution.get(f'X[{u}][{k}]', 0))
        print(f"u={u}: {row}")
        
    # print ordering of vertices based on the solution
    ordering = []
    for u in range(n):
        ordering.append(sum(best_solution.get(f'X[{u}][{k}]', 0) for k in range(n)))
    print(f"Ordering of vertices based on solution: {ordering}")
    
    print(minla.calculate_min_linear_arrangement(graph, ordering))
    
    