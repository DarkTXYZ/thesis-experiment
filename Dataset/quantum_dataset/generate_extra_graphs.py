import networkx as nx
import numpy as np
import random
import os
import sys
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from Baseline.lower_bound import MinLALowerBounds
from generate_dataset import generate_connected_random_graph, is_isomorphic_to_any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXTRA_SEED = 1042
DENSITY = 0.5
COUNTS = {10: 2, 15: 1, 20: 1, 25: 1}


def load_existing_graphs(n: int) -> list[nx.Graph]:
    filepath = os.path.join(SCRIPT_DIR, f"quantum_n{n}.pkl")
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    graphs = []
    for graph_data in data["graphs"]:
        G = nx.Graph()
        G.add_nodes_from(range(graph_data["num_vertices"]))
        G.add_edges_from(graph_data["edges"])
        graphs.append(G)
    return graphs


def generate_extra_graphs_for_n(n: int, count: int) -> list:
    existing_graphs = load_existing_graphs(n)
    new_graphs = []
    graphs_data = []

    random.seed(EXTRA_SEED + n)
    np.random.seed(EXTRA_SEED + n)

    i = 0
    attempts = 0
    max_attempts = count * 1000

    while i < count and attempts < max_attempts:
        graph = generate_connected_random_graph(n, DENSITY)

        if is_isomorphic_to_any(graph, existing_graphs) or is_isomorphic_to_any(graph, new_graphs):
            attempts += 1
            continue

        new_graphs.append(graph)

        lower_bound = MinLALowerBounds(graph).bound()

        graphs_data.append({
            "id": 100 + i,
            "num_vertices": n,
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "edges": list(graph.edges()),
            "lower_bound": lower_bound,
        })

        i += 1
        attempts += 1

    if i < count:
        print(f"  WARNING: Only generated {i}/{count} extra graphs for N={n} after {attempts} attempts")

    return graphs_data


def main():
    output_path = os.path.join(SCRIPT_DIR, "quantum_extra.pkl")
    extra_data = {}

    for n, count in COUNTS.items():
        print(f"Generating {count} extra graph(s) for N={n}...")
        graphs_data = generate_extra_graphs_for_n(n, count)
        extra_data[n] = {
            "num_vertices": n,
            "density": DENSITY,
            "num_graphs": len(graphs_data),
            "seed": EXTRA_SEED + n,
            "graphs": graphs_data,
        }
        print(f"  Generated {len(graphs_data)} graph(s) for N={n}")

    with open(output_path, "wb") as f:
        pickle.dump(extra_data, f)

    print(f"\nSaved extra graphs -> {output_path}")


if __name__ == "__main__":
    main()
