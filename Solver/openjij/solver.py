import openjij as oj
import random
from pyqubo import Array
import networkx as nx
import numpy as np

def generate_minla_qubo(G: nx.Graph):
    """Generate MINLA QUBO with optimized constraint construction."""
    N = G.number_of_nodes()
    _lambda = N
    
    X = Array.create('X', shape=(N, N), vartype='BINARY')
    
    # Thermometer constraint - optimized
    H_thermometer = _lambda * sum(
        (1 - X[u][k]) * X[u][k+1]
        for u in range(N)
        for k in range(N-1)
    )
    
    # Bijective constraint - optimized with list comprehension
    H_bijective = _lambda * sum(
        ((N - k) - sum(X[u][k] for u in range(N))) ** 2
        for k in range(N)
    )
    
    # Objective - use edges directly (no intermediate diff variable)
    H_objective = sum(
        sum(X[u][k] + X[v][k] - 2 * X[u][k] * X[v][k] for k in range(N))
        for u, v in G.edges
    )
    
    H = H_thermometer + H_bijective + H_objective
    model = H.compile()
    return model.to_bqm()

def generate_Q_matrix(bqm):
    """Generate Q matrix with optimized quadratic term extraction."""
    Q = {}
    offset = bqm.offset
    
    # Get linear terms directly
    for var, bias in bqm.linear.items():
        Q[(var, var)] = bias
    
    # Get quadratic terms directly (no exception handling in loop)
    for (qi, qj), bias in bqm.quadratic.items():
        Q[(qi, qj)] = bias
    
    return Q, offset

N = 25
G = nx.erdos_renyi_graph(N, 0.5, seed=42)
Q, offset = generate_Q_matrix(generate_minla_qubo(G))

# Solve with OpenJij.
sampler = oj.SQASampler()
response = sampler.sample_qubo(Q, num_reads=10)

def check_feasibility(sample: dict, N):
    """Check feasibility with optimized numpy operations and pre-computed variable names."""
    # Pre-compute variable names to avoid repeated string formatting
    var_names = [[f'X[{u}][{k}]' for k in range(N)] for u in range(N)]
    
    # Build solution matrix efficiently using list comprehension
    sol = np.array([[sample[var_names[u][k]] for k in range(N)] for u in range(N)], dtype=int)
    
    # Check thermometer constraint (vectorized where possible)
    for u in range(N):
        row = sol[u]
        # Check if any 0 is followed by 1
        if np.any((row[:-1] == 0) & (row[1:] == 1)):
            return False, "Thermometer constraint violated", sol
    
    # Check bijective constraint using numpy sum
    labels = np.sum(sol, axis=1)
    
    # Check if all labels are unique and in range [1, N]
    if len(np.unique(labels)) != N or not np.array_equal(np.sort(labels), np.arange(1, N+1)):
        return False, "Bijective constraint violated", sol
    
    return True, "All constraints satisfied", sol

print("Energy: ", response.first.energy + offset)
feasible, message, sol = check_feasibility(response.first.sample, N)
print("is_feasible: ", feasible, ", message: ", message)
print(sol)