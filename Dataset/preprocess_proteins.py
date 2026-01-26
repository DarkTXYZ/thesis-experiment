# Preprocess PROTEINS dataset to extract individual graphs
# Format: TU Dortmund graph dataset format
# Output: edge list files for top N graphs with most vertices (max 15 vertices)

import os
from collections import defaultdict

def preprocess_proteins(input_dir, output_dir, num_graphs=5, max_vertices=15):
    """
    Preprocess PROTEINS dataset and extract the top num_graphs graphs with most vertices.
    
    Args:
        input_dir: Directory containing PROTEINS files
        output_dir: Directory to save preprocessed graphs
        num_graphs: Number of graphs to extract (default 5)
        max_vertices: Maximum number of vertices allowed (default 15)
    """
    
    # Read graph indicator (maps node_id to graph_id)
    graph_indicator_path = os.path.join(input_dir, 'PROTEINS_graph_indicator.txt')
    node_to_graph = {}
    graph_node_count = defaultdict(int)
    
    print("Reading graph indicator...")
    with open(graph_indicator_path, 'r') as f:
        for node_id, line in enumerate(f, start=1):
            graph_id = int(line.strip())
            node_to_graph[node_id] = graph_id
            graph_node_count[graph_id] += 1
    
    print(f"Total nodes: {len(node_to_graph)}")
    print(f"Total graphs: {len(graph_node_count)}")
    
    # Filter graphs with <= max_vertices, then select top num_graphs with most vertices
    filtered_graphs = [(g_id, count) for g_id, count in graph_node_count.items() if count <= max_vertices]
    print(f"Graphs with <= {max_vertices} vertices: {len(filtered_graphs)}")
    
    sorted_by_nodes = sorted(filtered_graphs, key=lambda x: x[1], reverse=True)
    valid_graphs = set(g_id for g_id, _ in sorted_by_nodes[:num_graphs])
    print(f"Selecting top {num_graphs} graphs with most vertices (max {max_vertices}):")
    for g_id, count in sorted_by_nodes[:num_graphs]:
        print(f"  Graph {g_id}: {count} vertices")
    
    # Build node mapping for each valid graph (original node_id -> 0-indexed local node_id)
    graph_node_mapping = defaultdict(dict)
    for node_id, graph_id in node_to_graph.items():
        if graph_id in valid_graphs:
            local_id = len(graph_node_mapping[graph_id])
            graph_node_mapping[graph_id][node_id] = local_id
    
    # Read adjacency matrix and build edges for valid graphs
    adj_path = os.path.join(input_dir, 'PROTEINS_A.txt')
    graph_edges = defaultdict(set)
    
    print("Reading adjacency matrix...")
    with open(adj_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            node1, node2 = int(parts[0].strip()), int(parts[1].strip())
            
            graph_id = node_to_graph.get(node1)
            if graph_id and graph_id in valid_graphs:
                # Map to local 0-indexed node IDs
                local1 = graph_node_mapping[graph_id][node1]
                local2 = graph_node_mapping[graph_id][node2]
                
                # Store as undirected edge (smaller id first to avoid duplicates)
                edge = (min(local1, local2), max(local1, local2))
                if edge[0] != edge[1]:  # No self-loops
                    graph_edges[graph_id].add(edge)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each graph as edge list
    print(f"Saving graphs to {output_dir}...")
    saved_count = 0
    for graph_id in sorted(valid_graphs):
        n_nodes = graph_node_count[graph_id]
        edges = graph_edges[graph_id]
        n_edges = len(edges)
        
        if n_edges == 0:
            continue  # Skip graphs with no edges
            
        output_path = os.path.join(output_dir, f'PROTEINS_g{graph_id}_preprocessed.txt')
        with open(output_path, 'w') as f:
            # First line: n_nodes n_edges
            f.write(f"{n_nodes} {n_edges}\n")
            # Edge list
            for u, v in sorted(edges):
                f.write(f"{u} {v}\n")
        
        saved_count += 1
    
    print(f"Saved {saved_count} graphs")
    return saved_count


if __name__ == "__main__":
    input_dir = "Dataset/raw/PROTEINS"
    output_dir = "Dataset/processed/proteins"
    
    preprocess_proteins(input_dir, output_dir, num_graphs=5)
