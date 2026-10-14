import networkx as nx
import numpy as np
import random
import json
import os
import sys
import pickle
from pathlib import Path
from networkx.generators.atlas import graph_atlas_g

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Baseline.lower_bound import MinLALowerBounds


def generate_connected_random_graph(n: int, density: float, max_attempts: int = 10000) -> tuple[nx.Graph, int]:
    discarded = 0
    for _ in range(max_attempts):
        G = nx.gnp_random_graph(n, density)
        if nx.is_connected(G):
            return G, discarded
        discarded += 1

    # Fallback: connect components if max_attempts fail
    G = nx.gnp_random_graph(n, density)
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        comp1 = random.choice(list(components[0]))
        comp2 = random.choice(list(components[1]))
        G.add_edge(comp1, comp2)
        discarded += 1 # Count the initial fallback generation as a discarded attempt
        
    return G, discarded


def is_isomorphic_to_any(graph: nx.Graph, graph_list: list[nx.Graph]) -> bool:
    for existing_graph in graph_list:
        if nx.is_isomorphic(graph, existing_graph):
            return True
    return False


def generate_graphs_for_vertex_count(n: int, num_graphs: int, density: float, seed: int) -> list:
    graphs_data = []
    unique_graphs = []

    # Set seed once at the beginning
    random.seed(seed)
    np.random.seed(seed)

    i = 0
    attempts = 0
    total_discarded = 0
    max_attempts = num_graphs * 100  

    while i < num_graphs and attempts < max_attempts:
        graph, discarded_count = generate_connected_random_graph(n, density)
        total_discarded += discarded_count

        if is_isomorphic_to_any(graph, unique_graphs):
            attempts += 1
            continue

        unique_graphs.append(graph)

        lower_bound = MinLALowerBounds(graph).bound()

        graph_data = {
            'id': i,
            'num_vertices': n,
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'edges': list(graph.edges()),
            'lower_bound': lower_bound
        }
        graphs_data.append(graph_data)

        i += 1
        attempts += 1

        if i % 25 == 0:
            print(f"    Generated {i}/{num_graphs} unique graphs (attempts: {attempts})")

    if total_discarded > 0:
        print(f"    [REPORT] Discarded {total_discarded} disconnected graphs during generation.")
        
    if i < num_graphs:
        print(f"    WARNING: Only generated {i} unique graphs out of {num_graphs} requested after {attempts} attempts")

    return graphs_data


def generate_exhaustive_graphs(n: int) -> list:
    """All non-isomorphic connected simple graphs on n vertices.

    Uses the networkx graph atlas, which only covers graphs up to 7 nodes.
    """
    atlas = graph_atlas_g()
    connected = [g for g in atlas if g.number_of_nodes() == n and nx.is_connected(g)]

    graphs_data = []
    for i, graph in enumerate(connected):
        lower_bound = MinLALowerBounds(graph).bound()
        graphs_data.append({
            'id': i,
            'num_vertices': n,
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'edges': list(graph.edges()),
            'lower_bound': lower_bound,
        })

    return graphs_data


def save_exhaustive_graphs_to_pickle(n: int, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating all non-isomorphic connected graphs with {n} vertices (exhaustive)...")
    graphs_data = generate_exhaustive_graphs(n)

    data = {
        'num_vertices': n,
        'generation': 'exhaustive',
        'num_graphs': len(graphs_data),
        'graphs': graphs_data,
    }

    filename = f"quantum_n{n}.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    avg_edges = sum(g['num_edges'] for g in graphs_data) / len(graphs_data)
    avg_density = sum(g['density'] for g in graphs_data) / len(graphs_data)

    print(f"  Saved to {filename}")
    print(f"  Total unique graphs: {len(graphs_data)}")
    print(f"  Avg edges: {avg_edges:.1f}, Avg density: {avg_density:.4f}")

    return {
        'name': f"quantum_n{n}",
        'num_vertices': n,
        'num_graphs': len(graphs_data),
        'generation': 'exhaustive',
        'avg_density': avg_density,
        'avg_edges': avg_edges,
    }


def save_graphs_to_pickle(num_vertices_list: list, num_graphs: int, densities: list, seed: int, output_dir: str,
                           extra_results: list = None):
    os.makedirs(output_dir, exist_ok=True)

    all_results = list(extra_results) if extra_results else []

    for density in densities:
        for n in num_vertices_list:
            print(f"\nGenerating {num_graphs} graphs with {n} vertices (Density: {density})...")

            graphs_data = generate_graphs_for_vertex_count(n, num_graphs, density, seed)

            data = {
                'num_vertices': n,
                'density': density,
                'num_graphs': len(graphs_data),
                'seed': seed,
                'graphs': graphs_data
            }

            filename = f"quantum_n{n}_d{density}.pkl"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            avg_edges = sum(g['num_edges'] for g in graphs_data) / len(graphs_data)
            avg_density = sum(g['density'] for g in graphs_data) / len(graphs_data)

            print(f"  Saved to {filename}")
            print(f"  Total unique graphs: {len(graphs_data)}")
            print(f"  Avg edges: {avg_edges:.1f}, Avg density: {avg_density:.4f}")

            result = {
                'name': f"quantum_n{n}_d{density}",
                'num_vertices': n,
                'num_graphs': len(graphs_data),
                'density': density,
                'avg_density': avg_density,
                'avg_edges': avg_edges,
            }

            all_results.append(result)

    # Save summary JSON
    summary_file = os.path.join(output_dir, "minla_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved to {summary_file}")

    return all_results


def load_graphs(filepath: str) -> dict:
    """Load graphs from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    seed = 42
    exhaustive_n = 5                      # large_N5: complete enumeration, no density
    num_vertices_list = [10, 15, 20, 25]  # Updated to requested array
    num_graphs = 100
    densities = [0.2, 0.5, 0.8]           # Updated to handle multiple densities

    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Quantum Dataset Generator")
    print("=" * 60)
    print(f"Exhaustive vertex count: {exhaustive_n}")
    print(f"Vertex counts: {num_vertices_list}")
    print(f"Graphs per size: {num_graphs}")
    print(f"Edge densities: {densities}")
    print(f"Seed: {seed}")
    print("=" * 60)

    exhaustive_result = save_exhaustive_graphs_to_pickle(exhaustive_n, output_dir)
    results = save_graphs_to_pickle(num_vertices_list, num_graphs, densities, seed, output_dir,
                                     extra_results=[exhaustive_result])

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()