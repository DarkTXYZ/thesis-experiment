import networkx as nx
import sys
from typing import List
from pathlib import Path
import random


sys.path.append(str(Path(__file__).parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement

random.seed(42)

def init_ordering(graph: nx.Graph, method: str) -> List[int]:
    """Initialize ordering based on the specified method."""
    if method == "random":
        ordering = list(graph.nodes())
        random.shuffle(ordering)
        return ordering
    elif method == "normal":
        return list(graph.nodes())
    elif method == "bfs":
        random_start_node = random.choice(list(graph.nodes()))
        bfs_nodes = list(nx.bfs_tree(graph, source=random_start_node).nodes())
        return bfs_nodes
    elif method == "dfs":
        random_start_node = random.choice(list(graph.nodes()))
        dfs_nodes = list(nx.dfs_tree(graph, source=random_start_node).nodes())
        return dfs_nodes
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def increment(graph, label, i, ordering, x):
    label[ordering[i]] = x
    c = 0
    for j in range(i+1):
        if graph.has_edge(ordering[i], ordering[j]):
            c += abs(label[ordering[i]] - label[ordering[j]])
    return c

def successive_augmentation(graph: nx.Graph, method: str) -> List[int]:
    N = graph.number_of_nodes()
    ordering = init_ordering(graph, method)
    
    label = {node: 0 for node in graph.nodes()}
    l = -1
    r = 1
    
    for i, v in enumerate(ordering):
        if increment(graph, label, i, ordering, l) < increment(graph, label, i, ordering, r):
            label[v] = l
            l -= 1
        else:
            label[v] = r
            r += 1
    
    for v in ordering:
        label[v] -= l
        
    return [label[i] for i in range(N)]
    
if __name__ == "__main__":
    # Read file from Dataset/processed and create graph
    # Then calculate successive augmentation cost and store in csv file
    import csv
    from pathlib import Path
    
    # Get processed datasets directory
    dataset_dir = Path(__file__).parent.parent / "Dataset" / "processed"
    results_dir = Path(__file__).parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Output CSV file
    output_file = results_dir / "successive_augmentation_results.csv"
    
    # Find all preprocessed graph files
    graph_files = sorted(dataset_dir.glob("*_preprocessed.txt"))
    
    if not graph_files:
        print(f"No preprocessed graph files found in {dataset_dir}")
        exit(1)
    
    # Initialization methods to test
    methods = ["random", "normal", "bfs", "dfs"]
    
    print(f"Processing {len(graph_files)} graph files with {len(methods)} initialization methods...")
    print("=" * 80)
    
    # Store results
    results = []
    
    for graph_file in graph_files:
        print(f"\nProcessing: {graph_file.name}")
        
        try:
            # Read graph from file
            G = nx.Graph()
            with open(graph_file, 'r') as f:
                lines = f.readlines()
                
                # First line: n m
                n, m = map(int, lines[0].strip().split())
                
                # Add edges
                for line in lines[1:]:
                    if line.strip():
                        u, v = map(int, line.strip().split())
                        G.add_edge(u, v)
            
            print(f"  Vertices: {n}, Edges: {m}")
            
            # Test each initialization method
            for method in methods:
                try:
                    ordering = successive_augmentation(G, method)
                    cost = calculate_min_linear_arrangement(G, ordering)
                    
                    # Store result
                    results.append({
                        'dataset': graph_file.stem.replace('_preprocessed', ''),
                        'vertices': n,
                        'edges': m,
                        'method': method,
                        'cost': cost
                    })
                    
                    print(f"  Method '{method}': Cost = {cost}")
                    
                except Exception as e:
                    print(f"  Method '{method}': Error - {e}")
                    continue
            
        except Exception as e:
            print(f"  Error processing {graph_file.name}: {e}")
            continue
    
    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'vertices', 'edges', 'method', 'cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(results)} results successfully")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("Summary by Method:")
        for method in methods:
            method_results = [r for r in results if r['method'] == method]
            if method_results:
                avg_cost = sum(r['cost'] for r in method_results) / len(method_results)
                print(f"  {method}: {len(method_results)} datasets, Avg Cost = {avg_cost:.2f}")
    else:
        print("\nNo results to save")