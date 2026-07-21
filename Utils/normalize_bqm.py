from MinLA import generate_bqm_instance, decode_solution
import networkx as nx
import dimod
import matplotlib.pyplot as plt
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, PathIntegralAnnealingSampler

N = 25
G = nx.erdos_renyi_graph(N, 0.5)
bqm = generate_bqm_instance(G)
inv_scalar = bqm.normalize()
s = 1 / inv_scalar
    
# solver = PathIntegralAnnealingSampler()
solver = SimulatedAnnealingSampler()
sampleset = solver.sample(bqm, num_reads=10, beta_range=[0.1 * s, 100.0 * s])

for sample in sampleset.samples():
    decoded, feasible = decode_solution(sample, N)
    if feasible:
        print("Feasible solution found:", decoded)
    else:
        print("Infeasible solution found")