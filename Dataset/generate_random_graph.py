# given a list of number of vertices and an edge density, generate connected random graphs
# that has no self-loops and no duplicate edges
# output to processed folder with the correct format like in the processed folder

import random
import networkx as nx
from pathlib import Path

def generate_connected_random_graph(n, density, seed=None):
    """
    Generate a connected random graph with n vertices and given edge density.
    
    Args:
        n: Number of vertices
        density: Edge density (0 to 1), where 1 means complete graph
        seed: Random seed for reproducibility
        
    Returns:
        List of edges as (u, v) tuples
    """
    if seed is not None:
        random.seed(seed)
        
    # Calculate number of edges from density
    max_edges = n * (n - 1) // 2
    num_edges = int(density * max_edges)
    
    # Ensure minimum connectivity (at least n-1 edges for connected graph)
    num_edges = max(num_edges, n - 1)
    
    # Generate random connected graph using NetworkX
    G = nx.gnm_random_graph(n, num_edges, seed=seed)
    
    # Ensure graph is connected by adding edges if needed
    while not nx.is_connected(G):
        # Find disconnected components
        components = list(nx.connected_components(G))
        # Connect two random components
        comp1 = random.choice(list(components[0]))
        comp2 = random.choice(list(components[1]))
        G.add_edge(comp1, comp2)
    
    # Get edges
    edges = list(G.edges())
    
    return edges

def write_graph_file(filepath, n, edges):
    """Write graph to file in the standard format"""
    with open(filepath, 'w') as f:
        # Write header: n m
        f.write(f"{n} {len(edges)}\n")
        
        # Write edges
        for u, v in edges:
            f.write(f"{u} {v}\n")
    
    print(f"Created: {filepath.name} ({n} vertices, {len(edges)} edges)")

def main():
    # Configuration
    vertex_counts = [200, 300]
    densities = [0.5]
    
    # Get script directory and create processed folder
    script_dir = Path(__file__).parent
    processed_dir = script_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    print("Generating random connected graphs")
    print("=" * 60)
    print(f"Vertex counts: {vertex_counts}")
    print(f"Densities: {densities}")
    print(f"Output directory: {processed_dir}")
    print("=" * 60)
    print()
    
    graph_count = 0
    
    # Generate graphs for each combination
    for n in vertex_counts:
        for density in densities:
            # Generate graph
            edges = generate_connected_random_graph(n, density, seed=42 + graph_count)
            
            # Create filename
            density_str = str(int(density * 100))
            filename = f"random_n{n}_d{density_str}_preprocessed.txt"
            filepath = processed_dir / filename
            
            # Write to file
            write_graph_file(filepath, n, edges)
            
            graph_count += 1
    
    print()
    print("=" * 60)
    print(f"Generation complete: {graph_count} graphs created")

if __name__ == "__main__":
    main()

