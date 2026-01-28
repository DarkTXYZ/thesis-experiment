import networkx as nx
import pickle
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Baseline.local_search import local_search
from Baseline.spectral_sequencing import spectral_sequencing
from Baseline.successive_augmentation import successive_augmentation
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def read_edge_file(filepath: str) -> nx.Graph:
    """Read edge list file and create a graph. Handles multiple formats."""
    G = nx.Graph()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Check if first line is "num_vertices num_edges" format
        # This format has: first_number > 10 OR first_number == second_number + 1
        # (because num_vertices is usually > num_edges OR exactly num_edges+1 for a tree)
        first_line = lines[0].strip().split()
        start_idx = 0
        
        if len(first_line) == 2 and len(lines) > 1:
            try:
                first_num = int(first_line[0])
                second_num = int(first_line[1])
                
                # Heuristic: if first number is significantly larger (>10) 
                # AND there are enough following lines matching the edge count,
                # then it's likely a header
                if first_num > 10 and len(lines) - 1 >= second_num * 0.8:
                    # Likely a header line
                    start_idx = 1
                else:
                    # Likely an edge (0-indexed nodes can be small numbers)
                    start_idx = 0
            except ValueError:
                # First line is not parseable as numbers
                start_idx = 0
        
        # Read edges
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
    
    return G


def relabel_graph_nodes(G: nx.Graph) -> tuple[nx.Graph, bool]:
    """
    Relabel nodes to be consecutive starting from 0.
    Returns the relabeled graph and whether relabeling was needed.
    """
    nodes = sorted(G.nodes())
    
    # Check if already 0-indexed and consecutive
    is_already_correct = (nodes == list(range(len(nodes))))
    
    if is_already_correct:
        return G, False
    
    # Create mapping to 0-indexed consecutive
    mapping = {node: i for i, node in enumerate(nodes)}
    return nx.relabel_nodes(G, mapping), True


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


def process_graph_file(filepath: Path, graph_id: int) -> dict:
    """Process a single graph file and calculate baseline results."""
    print(f"  Processing: {filepath.name}")
    
    # Read and relabel graph
    G = read_edge_file(str(filepath))
    
    # Get original node range for debugging
    nodes = sorted(G.nodes())
    original_min = min(nodes) if nodes else 0
    original_max = max(nodes) if nodes else 0
    
    # Check if connected
    if not nx.is_connected(G):
        # Take largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"    Warning: Graph was disconnected. Using largest component with {len(G.nodes())} nodes")
    
    # Relabel nodes to 0-based consecutive
    G, was_relabeled = relabel_graph_nodes(G)
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if was_relabeled:
        print(f"    Vertices: {n}, Edges: {m} (relabeled from {original_min}-indexed)")
    else:
        print(f"    Vertices: {n}, Edges: {m} (already 0-indexed)")
    
    # Calculate baseline results
    baseline_results = calculate_minla_baseline(G)
    
    print(f"    Best cost: {baseline_results['best_cost']}")
    
    return {
        'id': graph_id,
        'name': filepath.stem,
        'num_vertices': n,
        'num_edges': m,
        'edges': list(G.edges()),
        'spectral_cost': baseline_results['spectral_cost'],
        'successive_augmentation_cost': baseline_results['successive_augmentation_cost'],
        'successive_augmentation_method': baseline_results['successive_augmentation_method'],
        'local_search_cost': baseline_results['local_search_cost'],
        'best_cost': baseline_results['best_cost']
    }


def main():
    """Preprocess real-world graphs and save to pickle file."""
    
    script_dir = Path(__file__).parent
    
    # Find all .edges files in current directory
    graph_files = sorted(script_dir.glob("*.edges"))
    
    # Also include .txt files (like iscas89-s27.txt)
    graph_files.extend(sorted(script_dir.glob("*.txt")))
    
    # Filter out this script file if it appears
    graph_files = [f for f in graph_files if f.name != 'preprocess_real_world.py']
    
    if not graph_files:
        print("No graph files found!")
        return
    
    print("=" * 60)
    print("Real-World Dataset Preprocessing")
    print("=" * 60)
    print(f"Found {len(graph_files)} graph files")
    for gf in graph_files:
        print(f"  - {gf.relative_to(script_dir)}")
    print("=" * 60)
    
    graphs_data = []
    
    for idx, graph_file in enumerate(graph_files):
        try:
            graph_data = process_graph_file(graph_file, idx)
            graphs_data.append(graph_data)
        except Exception as e:
            print(f"    Error processing {graph_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not graphs_data:
        print("\nNo graphs processed successfully!")
        return
    
    # Calculate statistics
    num_vertices = [g['num_vertices'] for g in graphs_data]
    num_edges = [g['num_edges'] for g in graphs_data]
    best_costs = [g['best_cost'] for g in graphs_data]
    
    avg_vertices = sum(num_vertices) / len(num_vertices)
    avg_edges = sum(num_edges) / len(num_edges)
    avg_best_cost = sum(best_costs) / len(best_costs)
    
    # Calculate approximate density
    densities = [2 * g['num_edges'] / (g['num_vertices'] * (g['num_vertices'] - 1)) 
                 for g in graphs_data if g['num_vertices'] > 1]
    avg_density = sum(densities) / len(densities) if densities else 0.0
    
    # Create data structure for pickle file
    data = {
        'dataset_type': 'real_world',
        'num_graphs': len(graphs_data),
        'avg_vertices': avg_vertices,
        'avg_edges': avg_edges,
        'avg_density': avg_density,
        'graphs': graphs_data
    }
    
    # Save to pickle file
    output_file = script_dir / "quantum_real_world.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed graphs: {len(graphs_data)}")
    print(f"Avg vertices: {avg_vertices:.1f}")
    print(f"Avg edges: {avg_edges:.1f}")
    print(f"Avg density: {avg_density:.3f}")
    print(f"Avg best cost: {avg_best_cost:.1f}")
    print(f"\nSaved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
