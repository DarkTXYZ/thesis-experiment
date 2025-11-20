# read all graph data from a directory that this file is in and preprocess them by reindexing them 
# and add two number at the first line: number of vertices and number of edges
# if input graph contains duplicate edges or self-loop, throw exception
# if input graph is not connected, throw exception

import os
from pathlib import Path

def read_edges_file(filepath):
    """Read .edges file format (simple edge list)"""
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
    return edges

def read_mtx_file(filepath):
    """Read MatrixMarket .mtx file format"""
    edges = []
    with open(filepath, 'r') as f:
        # Skip header lines
        for line in f:
            line = line.strip()
            if line.startswith('%'):
                continue
            if line.startswith('%%MatrixMarket'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
    return edges

def reindex_edges(edges):
    """Reindex vertices to be 0-indexed and consecutive"""
    # Check for self-loops and duplicate edges
    edge_set = set()
    for u, v in edges:
        # Check for self-loop
        if u == v:
            raise ValueError(f"Self-loop found: {u} - {v}")
        
        # Normalize edge (treat as undirected by sorting)
        edge_tuple = tuple(sorted([u, v]))
        if edge_tuple in edge_set:
            raise ValueError(f"Duplicate edge found: {u} - {v}")
        edge_set.add(edge_tuple)
    
    # Find all unique vertices
    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    
    # Check if graph is connected using Union-Find
    parent = {v: v for v in vertices}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y
    
    # Union all edges
    for u, v in edges:
        union(u, v)
    
    # Check if all vertices have the same root (connected)
    roots = set(find(v) for v in vertices)
    if len(roots) > 1:
        raise ValueError(f"Graph is not connected: {len(roots)} connected components found")
    
    # Create mapping from old to new indices
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(vertices))}
    
    # Reindex edges
    reindexed_edges = [(old_to_new[u], old_to_new[v]) for u, v in edges]
    
    num_vertices = len(vertices)
    num_edges = len(reindexed_edges)
    
    return num_vertices, num_edges, reindexed_edges

def write_preprocessed_file(filepath, num_vertices, num_edges, edges, output_dir):
    """Write preprocessed graph file"""
    output_path = output_dir / f"{filepath.stem}_preprocessed.txt"
    
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"{num_vertices} {num_edges}\n")
        
        # Write edges
        for u, v in edges:
            f.write(f"{u} {v}\n")
    
    print(f"Created: {output_path.name} ({num_vertices} vertices, {num_edges} edges)")
    return output_path

def preprocess_graph_file(filepath, output_dir):
    """Preprocess a single graph file"""
    try:
        # Read edges based on file extension
        if filepath.suffix == '.edges':
            edges = read_edges_file(filepath)
        elif filepath.suffix == '.mtx':
            edges = read_mtx_file(filepath)
        else:
            print(f"Skipping {filepath.name}: unsupported format")
            return None
        
        if not edges:
            print(f"Warning: {filepath.name} has no edges")
            return None
        
        # Reindex edges
        num_vertices, num_edges, reindexed_edges = reindex_edges(edges)
        
        # Write preprocessed file
        output_path = write_preprocessed_file(filepath, num_vertices, num_edges, reindexed_edges, output_dir)
        
        return output_path
        
    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
        return None

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    raw_dir = script_dir / "raw"
    processed_dir = script_dir / "processed"
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(exist_ok=True)
    
    print(f"Reading graph files from: {raw_dir}")
    print(f"Writing processed files to: {processed_dir}")
    print("=" * 60)
    
    # Check if raw directory exists
    if not raw_dir.exists():
        print(f"Error: 'raw' directory not found at {raw_dir}")
        print("Please create a 'raw' folder and place your graph files there.")
        return
    
    # Find all graph files in raw directory
    graph_files = []
    for ext in ['*.edges', '*.mtx']:
        graph_files.extend(raw_dir.glob(ext))
    
    if not graph_files:
        print("No graph files found (.edges or .mtx) in raw directory")
        return
    
    print(f"Found {len(graph_files)} graph file(s)\n")
    
    # Process each file
    processed_count = 0
    for filepath in sorted(graph_files):
        result = preprocess_graph_file(filepath, processed_dir)
        if result:
            processed_count += 1
    
    print("\n" + "=" * 60)
    print(f"Preprocessing complete: {processed_count}/{len(graph_files)} files processed")

if __name__ == "__main__":
    main()
