import networkx as nx
import numpy as np
import random
import json
import os
import sys
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


def generate_unique_graphs(num_vertices_list: list, seed: int) -> list:
    """
    Generate one unique random graph for each vertex count in the list.
    Returns list of (graph, name) tuples.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    graphs = []
    
    for n in num_vertices_list:
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            attempts += 1
            
            # Random density between 0.3 and 0.8
            density = random.uniform(0.3, 0.8)
            
            # Generate random graph
            graph = nx.erdos_renyi_graph(n, density, seed=seed + n + attempts)
            
            # Skip disconnected graphs
            if not nx.is_connected(graph):
                continue
            
            # Found a connected graph
            graph_name = f"quantum_n{n}"
            graphs.append((graph, graph_name))
            break
    
    return graphs


def save_graphs_with_minla(graphs: list, output_dir: str):
    """
    Save graphs and their MinLA costs to files.
    Creates:
    - Individual graph files (.edgelist) - NetworkX compatible
    - Summary JSON file with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for graph, name in graphs:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        # Calculate MinLA using baseline algorithms
        baseline_results = calculate_minla_baseline(graph)
        
        # Save graph as edge list (NetworkX compatible format)
        graph_file = os.path.join(output_dir, f"{name}.edgelist")
        nx.write_edgelist(graph, graph_file, data=False)
        
        # Store result
        result = {
            'name': name,
            'num_vertices': n,
            'num_edges': m,
            'spectral_cost': baseline_results['spectral_cost'],
            'successive_augmentation_cost': baseline_results['successive_augmentation_cost'],
            'successive_augmentation_method': baseline_results['successive_augmentation_method'],
            'local_search_cost': baseline_results['local_search_cost'],
            'best_cost': baseline_results['best_cost'],
            'edges': [list(e) for e in graph.edges()]
        }
        results.append(result)
        
        print(f"{name}: n={n}, m={m}")
        print(f"  Spectral: {baseline_results['spectral_cost']}, "
              f"SA ({baseline_results['successive_augmentation_method']}): {baseline_results['successive_augmentation_cost']}, "
              f"Local Search: {baseline_results['local_search_cost']}, "
              f"Best: {baseline_results['best_cost']}")
    
    # Save summary JSON
    summary_file = os.path.join(output_dir, "minla_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved {len(graphs)} graphs to {output_dir}")
    print(f"Summary saved to {summary_file}")
    
    return results


def load_graph(filepath: str) -> nx.Graph:
    """Load a graph from .edgelist file."""
    return nx.read_edgelist(filepath, nodetype=int)


def main():
    # Configuration
    seed = 42
    num_vertices_list = [5, 7, 10, 12]
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Generating graphs with {num_vertices_list} vertices (seed={seed})")
    print("-" * 50)
    
    # Generate unique graphs
    graphs = generate_unique_graphs(num_vertices_list, seed)
    
    # Save graphs and calculate MinLA
    results = save_graphs_with_minla(graphs, output_dir)
    
    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
