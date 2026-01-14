import networkx as nx
import numpy as np
import itertools
import random
import json
import os


def calculate_minla_bruteforce(graph: nx.Graph) -> tuple[int, list]:
    """
    Calculate the minimum linear arrangement using brute force.
    Returns (min_cost, best_ordering)
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    
    min_cost = float('inf')
    best_ordering = None
    
    # Try all permutations
    for perm in itertools.permutations(range(n)):
        # perm[i] gives the position of node i
        cost = 0
        for u, v in graph.edges():
            cost += abs(perm[u] - perm[v])
        
        if cost < min_cost:
            min_cost = cost
            best_ordering = list(perm)
    
    return min_cost, best_ordering


def is_isomorphic_to_any(graph: nx.Graph, graph_list: list) -> bool:
    """Check if graph is isomorphic to any graph in the list."""
    for existing_graph, _ in graph_list:
        if nx.is_isomorphic(graph, existing_graph):
            return True
    return False


def generate_unique_graphs(num_vertices_list: list, num_graphs: int, seed: int) -> list:
    """
    Generate unique non-isomorphic random graphs with specified vertex counts.
    Returns list of (graph, name) tuples.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    graphs = []
    
    graph_id = 0
    attempts = 0
    max_attempts = 1000
    
    while len(graphs) < num_graphs and attempts < max_attempts:
        attempts += 1
        
        # Randomly select number of vertices
        n = random.choice(num_vertices_list)
        
        # Random density between 0.3 and 0.8
        density = random.uniform(0.3, 0.8)
        
        # Generate random graph
        graph = nx.erdos_renyi_graph(n, density, seed=seed + attempts)
        
        # Skip disconnected graphs
        if not nx.is_connected(graph):
            continue
        
        # Check for isomorphism with existing graphs
        if not is_isomorphic_to_any(graph, graphs):
            graph_name = f"exact_n{n}_id{graph_id}"
            graphs.append((graph, graph_name))
            graph_id += 1
    
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
        
        # Calculate MinLA using brute force
        min_cost, best_ordering = calculate_minla_bruteforce(graph)
        
        # Save graph as edge list (NetworkX compatible format)
        graph_file = os.path.join(output_dir, f"{name}.edgelist")
        nx.write_edgelist(graph, graph_file, data=False)
        
        # Store result
        result = {
            'name': name,
            'num_vertices': n,
            'num_edges': m,
            'minla_cost': min_cost,
            'best_ordering': best_ordering,
            'edges': [list(e) for e in graph.edges()]
        }
        results.append(result)
        
        print(f"{name}: n={n}, m={m}, MinLA={min_cost}, ordering={best_ordering}")
    
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
    num_graphs = 7
    num_vertices_list = [3, 4]
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Generating {num_graphs} unique graphs with {num_vertices_list} vertices (seed={seed})")
    print("-" * 50)
    
    # Generate unique graphs
    graphs = generate_unique_graphs(num_vertices_list, num_graphs, seed)
    
    # Save graphs and calculate MinLA
    results = save_graphs_with_minla(graphs, output_dir)
    
    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
