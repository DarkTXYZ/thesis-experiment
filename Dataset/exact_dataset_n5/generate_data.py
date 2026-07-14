import networkx as nx
import itertools
import os
from pathlib import Path


def generate_all_non_isomorphic_graphs(num_nodes):
    """
    Generate all non-isomorphic graphs for a given number of nodes.
    Uses strict isomorphism checking to ensure no duplicates.
    
    Args:
        num_nodes: Number of vertices
    
    Returns:
        List of non-isomorphic graphs (guaranteed no isomorphic duplicates)
    """
    nodes = list(range(num_nodes))
    all_possible_edges = list(itertools.combinations(nodes, 2))
    
    non_isomorphic_graphs = []
    
    # Generate all possible subsets of edges
    for num_edges in range(len(all_possible_edges) + 1):
        for edges in itertools.combinations(all_possible_edges, r=num_edges):
            # Create graph from edges
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            # Check if isomorphic to any existing graph
            is_duplicate = False
            for existing_G in non_isomorphic_graphs:
                if nx.is_isomorphic(G, existing_G):
                    is_duplicate = True
                    break
            
            # If not a duplicate, add to list
            if not is_duplicate:
                non_isomorphic_graphs.append(G)
    
    return non_isomorphic_graphs


def generate_all_non_isomorphic_graphs_optimized(num_nodes):
    """
    Optimized version: Generate all non-isomorphic connected graphs with strict isomorphism checking.
    """
    nodes = list(range(num_nodes))
    all_possible_edges = list(itertools.combinations(nodes, 2))
    
    non_isomorphic_graphs = []
    
    # Generate all possible subsets of edges
    for num_edges in range(len(all_possible_edges) + 1):
        for edges in itertools.combinations(all_possible_edges, r=num_edges):
            # Create graph from edges
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            # Only consider connected graphs
            if not nx.is_connected(G):
                continue
            
            # Check if isomorphic to any existing graph
            is_duplicate = False
            for existing_G in non_isomorphic_graphs:
                if nx.is_isomorphic(G, existing_G):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                non_isomorphic_graphs.append(G)
    
    return non_isomorphic_graphs


def save_graphs_to_edgelist(graphs, num_nodes, output_dir):
    """
    Save graphs to edgelist format files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, G in enumerate(graphs):
        filename = os.path.join(output_dir, f"graph_n{num_nodes}_{idx}.edgelist")
        nx.write_edgelist(G, filename)
    
    print(f"Saved {len(graphs)} graphs for n={num_nodes} vertices")


def verify_no_isomorphic_duplicates(graphs):
    """
    Verify that the list of graphs contains no isomorphic duplicates.
    Returns True if all graphs are non-isomorphic, False otherwise.
    """
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            if nx.is_isomorphic(graphs[i], graphs[j]):
                print(f"  WARNING: Graphs {i} and {j} are isomorphic!")
                return False
    return True


def main():
    output_base = "/Users/pawaret/Desktop/Master/Experiment/Dataset/exact_dataset_n5"
    
    for num_nodes in [3, 4, 5]:
        print(f"\nGenerating all non-isomorphic graphs for {num_nodes} vertices...")
        graphs = generate_all_non_isomorphic_graphs_optimized(num_nodes)
        print(f"Found {len(graphs)} non-isomorphic graphs for n={num_nodes}")
        
        # Verify no isomorphic duplicates
        print(f"Verifying no isomorphic duplicates...")
        if verify_no_isomorphic_duplicates(graphs):
            print(f"✓ All {len(graphs)} graphs are confirmed non-isomorphic")
        else:
            print(f"✗ ERROR: Found isomorphic duplicates!")
        
        # Save to directory
        output_dir = os.path.join(output_base, f"n{num_nodes}")
        save_graphs_to_edgelist(graphs, num_nodes, output_dir)
        
        # Print summary
        print(f"  - Saved to: {output_dir}")
        for i, G in enumerate(graphs[:5]):  # Show first 5
            print(f"  - Graph {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, degree_seq={sorted(dict(G.degree()).values(), reverse=True)}")
        if len(graphs) > 5:
            print(f"  - ... and {len(graphs) - 5} more graphs")


if __name__ == "__main__":
    main()
