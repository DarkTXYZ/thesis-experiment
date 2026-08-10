import networkx as nx
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Utils.MinLA import generate_bqm_instance, decode_solution, calculate_min_linear_arrangement
import numpy as np

N = 25
G = nx.erdos_renyi_graph(N, 0.5)
highest_degree = max(dict(G.degree()).values())

bqm = generate_bqm_instance(G)
print(bqm)

lin = np.array(list(bqm.linear.values()))
quad = np.array(list(bqm.quadratic.values()))
all_coeffs = np.concatenate([lin, quad])
all_coeffs.sort()

# factor = bqm.normalize()

print("All coefficients:", all_coeffs)
print("Number of positive coefficients:", np.sum(all_coeffs > 0))
print("Highest scalar of positive coefficients:", all_coeffs[all_coeffs > 0].max())
print("Number of negative coefficients:", np.sum(all_coeffs < 0))
print("Lowest scalar of negative coefficients:", all_coeffs[all_coeffs < 0].min())
# print(-(2 * N * N - 2 * N + 1))

print("Variables that have lowest scalar of negative coefficients:", [var for var, coeff in bqm.linear.items() if coeff == all_coeffs[all_coeffs < 0].min()])



print("Number of zero coefficients:", N*N*(N*N+1)/2 - len(all_coeffs))

# print(13 - highest_degree)
