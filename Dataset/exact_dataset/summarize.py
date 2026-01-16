# for each graph, calculate ground truth MinLA using exhaustive search
# and store results along with graph data

import os
import json
import networkx as nx
from itertools import permutations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def load_graph(filepath: str) -> nx.Graph:
    """Load a graph from .edgelist file."""
    return nx.read_edgelist(filepath, nodetype=int)


def exhaustive_minla(graph: nx.Graph) -> tuple[int, list]:
    """
    Find the exact MinLA cost using exhaustive search over all permutations.
    
    Returns:
        (min_cost, best_ordering)
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    
    min_cost = float('inf')
    best_ordering = None
    
    for perm in permutations(range(n)):
        # Create ordering: ordering[node] = position
        ordering = [0] * n
        for pos, node in enumerate(perm):
            ordering[node] = pos
        
        cost = calculate_min_linear_arrangement(graph, ordering)
        
        if cost < min_cost:
            min_cost = cost
            best_ordering = list(ordering)
    
    return min_cost, best_ordering


def summarize_dataset():
    """Generate summary for all edgelist files in the exact_dataset directory."""
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    
    results = []
    
    # Find all .edgelist files
    edgelist_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.edgelist')])
    
    print(f"Found {len(edgelist_files)} edgelist files")
    print("=" * 50)
    
    for filename in edgelist_files:
        filepath = os.path.join(dataset_dir, filename)
        graph_name = filename.replace('.edgelist', '')
        
        # Load graph
        graph = load_graph(filepath)
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        print(f"Processing {graph_name}: n={n}, m={m}...")
        
        # Calculate exact MinLA
        min_cost, best_ordering = exhaustive_minla(graph)
        
        # Store result
        result = {
            'name': graph_name,
            'num_vertices': n,
            'num_edges': m,
            'minla_cost': min_cost,
            'best_ordering': best_ordering,
            'edges': [list(e) for e in graph.edges()]
        }
        results.append(result)
        
        print(f"  MinLA cost: {min_cost}")
        print(f"  Best ordering: {best_ordering}")
    
    # Save to JSON
    output_file = os.path.join(dataset_dir, 'minla_summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 50)
    print(f"Summary saved to {output_file}")
    print(f"Total graphs: {len(results)}")
    
    return results


if __name__ == "__main__":
    summarize_dataset()