"""
Enumerate all binary variable assignments of the MinLA QUBO formulation
(thermometer + bijective encoding, see Utils/MinLA.py) for every
non-isomorphic simple graph on N=3,4 vertices.

For each assignment X[u][k], prints:
  - cost term      : H_objective (numeric, depends only on the graph's edges)
  - constraint term: mu_thermometer * H_thermometer + mu_bijective * H_bijective
                      (left symbolic -- mu_thermometer / mu_bijective are NOT
                      substituted with numeric values)

N=5 is excluded: 2^25 assignments per graph is infeasible to enumerate/print.
"""

import csv
import itertools
import os

import networkx as nx

N_VALUES = [4]
CONSOLE_ROW_LIMIT = 0  # print in full up to this many rows/graph; beyond that, write CSV + preview
RESULTS_DIR = "Results/qubo_enumeration"


def non_isomorphic_graphs(n):
    """All non-isomorphic connected simple graphs on n labeled vertices (0..n-1)."""
    possible_edges = list(itertools.combinations(range(n), 2))
    graphs = []
    for r in range(len(possible_edges) + 1):
        for edge_subset in itertools.combinations(possible_edges, r):
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edge_subset)
            if not nx.is_connected(G):
                continue
            if not any(nx.is_isomorphic(G, H) for H in graphs):
                graphs.append(G)
    return graphs


def cost_term(X, edges, n):
    """H_objective = sum_{(u,v) in E} sum_k (X[u][k] + X[v][k] - 2*X[u][k]*X[v][k])"""
    total = 0
    for u, v in edges:
        for k in range(n):
            total += X[u][k] + X[v][k] - 2 * X[u][k] * X[v][k]
    return total


def constraint_coefficients(X, n):
    """Coefficients (c_thermo, c_bij) of mu_thermometer and mu_bijective."""
    c_thermo = 0
    for u in range(n):
        for k in range(n - 1):
            c_thermo += (1 - X[u][k]) * X[u][k + 1]

    c_bij = 0
    for k in range(n):
        col_sum = sum(X[u][k] for u in range(n))
        c_bij += ((n - k) - col_sum) ** 2

    return c_thermo, c_bij


def format_constraint(c_thermo, c_bij):
    terms = []
    if c_thermo:
        terms.append(f"{c_thermo}*mu_thermometer")
    if c_bij:
        terms.append(f"{c_bij}*mu_bijective")
    return " + ".join(terms) if terms else "0"


def format_assignment(X, n):
    return "|".join("".join(str(X[u][k]) for k in range(n)) for u in range(n))


def enumerate_assignments(graph, n):
    edges = list(graph.edges())
    rows = []
    for bits in itertools.product((0, 1), repeat=n * n):
        X = [bits[u * n:(u + 1) * n] for u in range(n)]
        cost = cost_term(X, edges, n)
        c_thermo, c_bij = constraint_coefficients(X, n)
        rows.append({
            "assignment": format_assignment(X, n),
            "cost_term": cost,
            "constraint_term": format_constraint(c_thermo, c_bij),
            "mu_thermometer_coeff": c_thermo,
            "mu_bijective_coeff": c_bij,
        })
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for n in N_VALUES:
        graphs = non_isomorphic_graphs(n)
        print(f"\n{'=' * 70}")
        print(f"N = {n}  |  {len(graphs)} non-isomorphic graph(s)  |  "
              f"{2 ** (n * n)} assignments per graph")
        print(f"{'=' * 70}")

        for gi, G in enumerate(graphs):
            edges = list(G.edges())
            rows = enumerate_assignments(G, n)
            print(f"\n--- Graph {gi + 1}/{len(graphs)}  edges={edges}  "
                  f"({len(rows)} assignments) ---")

            if len(rows) <= CONSOLE_ROW_LIMIT:
                for row in rows:
                    print(f"  X=[{row['assignment']}]  "
                          f"cost={row['cost_term']}  "
                          f"constraint={row['constraint_term']}")
            else:
                csv_path = os.path.join(RESULTS_DIR, f"N{n}_graph{gi + 1}.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  {len(rows)} rows exceed console limit "
                      f"({CONSOLE_ROW_LIMIT}) -> written to {csv_path}")
                for row in rows[:10]:
                    print(f"  X=[{row['assignment']}]  "
                          f"cost={row['cost_term']}  "
                          f"constraint={row['constraint_term']}")
                print(f"  ... ({len(rows) - 10} more rows in {csv_path})")


if __name__ == "__main__":
    main()
