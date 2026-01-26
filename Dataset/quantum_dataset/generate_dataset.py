import networkx as nx
import numpy as np
import random
import json
import os
import sys
import pickle
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Baseline.local_search import local_search
from Baseline.spectral_sequencing import spectral_sequencing
from Baseline.successive_augmentation import successive_augmentation
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def calculate_minla_baseline(graph: nx.Graph) -> dict:
    """
    Calculate MinLA costs using baseline algorithms.
    Returns dict with costs from each algorithm.
    """
    n = graph.number_of_nodes()
    
    # Spectral sequencing
    spectral_ordering = spectral_sequencing(graph)
    spectral_cost = calculate_min_linear_arrangement(graph, spectral_ordering)
    
    # Successive augmentation (using different initialization methods)
    sa_costs = {}
    for method in ["random", "normal", "bfs", "dfs"]:
        sa_ordering = successive_augmentation(graph, method)
        sa_costs[method] = calculate_min_linear_arrangement(graph, sa_ordering)
    best_sa_method = min(sa_costs, key=sa_costs.get)
    best_sa_cost = sa_costs[best_sa_method]
    
    # Local search (starting from spectral ordering)
    local_search_cost = local_search(graph, spectral_ordering, iter_max=1000, flip_method="flip2")
    
    return {
        'spectral_cost': spectral_cost,
        'successive_augmentation_cost': best_sa_cost,
        'successive_augmentation_method': best_sa_method,
        'local_search_cost': local_search_cost,
        'best_cost': min(spectral_cost, best_sa_cost, local_search_cost)
    }


def generate_connected_random_graph(n: int, density: float, max_attempts: int = 1000) -> nx.Graph:
    """
    Generate a connected random graph with n vertices and approximate edge density.
    """
    for _ in range(max_attempts):
        G = nx.gnp_random_graph(n, density)
        if nx.is_connected(G):
            return G
    
    # Fallback: connect components
    G = nx.gnp_random_graph(n, density)
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        comp1 = random.choice(list(components[0]))
        comp2 = random.choice(list(components[1]))
        G.add_edge(comp1, comp2)
    return G


def is_isomorphic_to_any(graph: nx.Graph, graph_list: list[nx.Graph]) -> bool:
    """
    Check if graph is isomorphic to any graph in the list.
    """
    for existing_graph in graph_list:
        if nx.is_isomorphic(graph, existing_graph):
            return True
    return False


def generate_graphs_for_vertex_count(n: int, num_graphs: int, density: float, seed: int) -> list:
    """
    Generate multiple unique non-isomorphic random graphs for a given vertex count.
    Returns list of graphs with their baseline results.
    """
    graphs_data = []
    unique_graphs = []  # Store nx.Graph objects to check isomorphism
    
    i = 0
    attempts = 0
    max_attempts = num_graphs * 100  # Prevent infinite loops
    
    while i < num_graphs and attempts < max_attempts:
        # Set seed for reproducibility
        graph_seed = seed + n * 1000 + attempts
        random.seed(graph_seed)
        np.random.seed(graph_seed)
        
        # Generate connected random graph
        graph = generate_connected_random_graph(n, density)
        
        # Check if isomorphic to any existing graph
        if is_isomorphic_to_any(graph, unique_graphs):
            attempts += 1
            continue
        
        # Graph is unique, add it to the list
        unique_graphs.append(graph)
        
        # Calculate MinLA using baseline algorithms
        baseline_results = calculate_minla_baseline(graph)
        
        graph_data = {
            'id': i,
            'num_edges': graph.number_of_edges(),
            'edges': list(graph.edges()),
            'spectral_cost': baseline_results['spectral_cost'],
            'successive_augmentation_cost': baseline_results['successive_augmentation_cost'],
            'successive_augmentation_method': baseline_results['successive_augmentation_method'],
            'local_search_cost': baseline_results['local_search_cost'],
            'best_cost': baseline_results['best_cost']
        }
        graphs_data.append(graph_data)
        
        i += 1
        attempts += 1
        
        if i % 10 == 0:
            print(f"    Generated {i}/{num_graphs} unique graphs (attempts: {attempts})")
    
    if i < num_graphs:
        print(f"    WARNING: Only generated {i} unique graphs out of {num_graphs} requested after {attempts} attempts")
    
    return graphs_data


def save_graphs_to_pickle(num_vertices_list: list, num_graphs: int, density: float, seed: int, output_dir: str):
    """
    Generate and save graphs to pickle files.
    Creates one pickle file per vertex count containing all graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for n in num_vertices_list:
        print(f"\nGenerating {num_graphs} graphs with {n} vertices...")
        
        graphs_data = generate_graphs_for_vertex_count(n, num_graphs, density, seed)
        
        # Create data structure for pickle file
        data = {
            'num_vertices': n,
            'density': density,
            'num_graphs': num_graphs,
            'seed': seed,
            'graphs': graphs_data
        }
        
        # Save to pickle file
        filename = f"quantum_n{n}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Calculate statistics
        avg_edges = sum(g['num_edges'] for g in graphs_data) / len(graphs_data)
        avg_best_cost = sum(g['best_cost'] for g in graphs_data) / len(graphs_data)
        
        print(f"  Saved to {filename}")
        print(f"  Total unique graphs: {len(graphs_data)}")
        print(f"  Avg edges: {avg_edges:.1f}, Avg best cost: {avg_best_cost:.1f}")
        
        # Store summary for JSON
        result = {
            'name': f"quantum_n{n}",
            'num_vertices': n,
            'num_graphs': num_graphs,
            'density': density,
            'avg_edges': avg_edges,
            'avg_best_cost': avg_best_cost
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
    # Configuration
    seed = 42
    num_vertices_list = [6, 8, 11, 13, 15]
    num_graphs = 50  # Number of graphs per vertex count
    density = 0.5  # Edge density
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("Quantum Dataset Generator")
    print("=" * 60)
    print(f"Vertex counts: {num_vertices_list}")
    print(f"Graphs per size: {num_graphs}")
    print(f"Edge density: {density}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Generate and save graphs
    results = save_graphs_to_pickle(num_vertices_list, num_graphs, density, seed, output_dir)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
