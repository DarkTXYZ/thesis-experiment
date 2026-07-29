"""
count_penalty.py
=================
For each graph in quantum_extra.pkl, estimate the maximum objective gain
obtainable by violating Utils.MinLA.generate_bqm_instance's constraints
(thermometer + bijective) by a single bit flip away from a feasible,
thermometer-encoded permutation - i.e. a sampling estimate of the minimal
sufficient penalty weight lambda* for this problem's actual BQM, instead of
a synthetic Q/c/A/b demo.

Sufficiency condition:
    mu > lambda*  :=  max_{y : P(y)>0}  ( f_ref - f_obj(y) ) / P(y)
where f_obj/P are dimod BQM energies of the RAW (unweighted) objective term
and the combined constraint term (H_thermometer + H_bijective, both without
their mu_thermo/mu_bijec penalty weight), f_ref is the best objective found
among sampled feasible permutations, and y ranges over every single-bit-flip
neighbour of those permutations.

Exhaustive enumeration is infeasible here (n^2 binary variables per graph,
so 2^100 states for the smallest graph in this dataset) - this is therefore
a sampling *lower bound* on lambda*, not an exact certificate. More samples
tighten (raise) the estimate; it can only ever be too small, never too large.
"""
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd
from pyqubo import Array

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Utils.MinLA import get_penalties

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset/quantum_extra.pkl")
CSV_PATH = os.path.join(PARENT_DIR, "penalty_gain_summary.csv")

N_FEASIBLE_SAMPLES = 1000
SEED = 0


def convert_graph_data_to_nx(graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["num_vertices"]))
    G.add_edges_from(graph_data["edges"])
    return G


def read_extra_graphs():
    with open(DATASET_PATH, "rb") as f:
        extra_data = pickle.load(f)

    graphs = []
    for group in extra_data.values():
        for graph_data in group["graphs"]:
            graphs.append(graph_data)
    return graphs


def build_raw_bqms(graph: nx.Graph):
    """H_thermometer/H_bijective WITHOUT the mu penalty weight, plus
    H_objective, each compiled to its own BQM - mirrors the term
    definitions in Utils.MinLA.generate_bqm_instance, but keeps every
    term separate instead of summing and weighting them."""
    n = graph.number_of_nodes()
    X = Array.create('X', shape=(n, n), vartype='BINARY')

    H_thermometer = sum(
        (1 - X[u][k]) * X[u][k + 1]
        for u in range(n)
        for k in range(n - 1)
    )
    H_bijective = sum(
        ((n - k) - sum(X[u][k] for u in range(n))) ** 2
        for k in range(n)
    )
    H_objective = sum(
        sum(X[u][k] + X[v][k] - 2 * X[u][k] * X[v][k] for k in range(n))
        for u, v in graph.edges
    )

    return (
        H_thermometer.compile().to_bqm(),
        H_bijective.compile().to_bqm(),
        H_objective.compile().to_bqm(),
    )


def rank_to_sample(rank, n):
    """rank[u] in {1,...,n}, a bijection (a permutation's 1-indexed
    position for each node). Builds the thermometer encoding
    X[u][k] = 1 iff k < rank[u] - the convention forced by
    Utils.MinLA.check_feasibility/decode_solution (row sum = position,
    column k holds exactly n-k ones)."""
    sample = {}
    for u in range(n):
        r = rank[u]
        for k in range(n):
            sample[f'X[{u}][{k}]'] = 1 if k < r else 0
    return sample


def delta_energy(bqm, sample, var):
    """Change in bqm.energy(sample) from flipping a single BINARY variable,
    computed from local adjacency instead of a full energy() recompute.
    H_bijective alone can have O(n^3) quadratic terms, and evaluating every
    single-bit-flip neighbour of hundreds of feasible samples from scratch
    would be far too slow - this is O(degree(var)) instead."""
    old = sample[var]
    new = 1 - old
    delta = bqm.linear[var] * (new - old)
    for neighbor, coeff in bqm.adj[var].items():
        delta += coeff * sample[neighbor] * (new - old)
    return delta


def estimate_penalty_gain(graph, n_samples=N_FEASIBLE_SAMPLES, seed=SEED):
    n = graph.number_of_nodes()
    thermometer_bqm, bijective_bqm, objective_bqm = build_raw_bqms(graph)
    all_vars = [f'X[{u}][{k}]' for u in range(n) for k in range(n)]

    rng = np.random.default_rng(seed)

    f_ref = np.inf
    feasible = []
    for _ in range(n_samples):
        rank = {u: int(r) for u, r in enumerate(rng.permutation(n) + 1)}
        sample = rank_to_sample(rank, n)
        p = thermometer_bqm.energy(sample) + bijective_bqm.energy(sample)
        assert abs(p) < 1e-6, f"constructed permutation should be feasible, got P={p}"
        f = objective_bqm.energy(sample)
        f_ref = min(f_ref, f)
        feasible.append((sample, f))

    best_ratio = -np.inf
    best_gain_once = -np.inf
    p_min = np.inf
    n_moves = 0

    for sample, f_base in feasible:
        for var in all_vars:
            dp = delta_energy(thermometer_bqm, sample, var) + delta_energy(bijective_bqm, sample, var)
            if dp <= 1e-9:
                continue
            df = delta_energy(objective_bqm, sample, var)
            gain = f_ref - (f_base + df)
            ratio = gain / dp
            n_moves += 1
            if ratio > best_ratio:
                best_ratio = ratio
            if dp < p_min - 1e-9:
                p_min, best_gain_once = dp, gain
            elif abs(dp - p_min) < 1e-9:
                best_gain_once = max(best_gain_once, gain)

    return dict(
        f_ref=f_ref,
        lambda_hat=best_ratio,
        gain_once_hat=best_gain_once,
        p_min=p_min,
        n_feasible=len(feasible),
        n_moves=n_moves,
    )


def main():
    rows = []
    for graph_data in read_extra_graphs():
        G = convert_graph_data_to_nx(graph_data)
        n = G.number_of_nodes()
        mu_thermo, mu_bijec = get_penalties(G)

        result = estimate_penalty_gain(G)
        rows.append(dict(
            graph_id=graph_data["id"],
            n=n,
            mu_lucas=mu_thermo,
            **result,
        ))

        print(f"graph {graph_data['id']:<6} n={n:<4} "
              f"mu_lucas={mu_thermo:<8.1f} "
              f"lambda_hat={result['lambda_hat']:<10.3f} "
              f"gain_once={result['gain_once_hat']:<10.3f} "
              f"p_min={result['p_min']:<8.3f} "
              f"n_moves={result['n_moves']}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nResults saved to {CSV_PATH}")
    return df


if __name__ == "__main__":
    main()
