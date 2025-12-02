import networkx as nx
import random
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement

def local_search(G : nx.Graph, ordering: list, iter_max: int = 1000, flip_method: str = "flip2") -> list:
    iter_count = 0
    
    def flip2():
        # randomly choose 2 vertices from G
        u, v = random.sample(list(G.nodes()), 2)
        new_ordering = ordering.copy()
        
        t = new_ordering[u]
        new_ordering[u] = new_ordering[v]
        new_ordering[v] = t
        
        return new_ordering
    
    def flip3():
        # randomly choose 3 vertices from G
        u, v, w = random.sample(list(G.nodes()), 3)
        new_ordering = ordering.copy()
        
        t = new_ordering[u]
        new_ordering[u] = new_ordering[v]
        new_ordering[v] = new_ordering[w]
        new_ordering[w] = t
        
        return new_ordering
    
    def flipEdge():
        # randomly choose an edge from G
        u, v = random.choice(list(G.edges()))
        new_ordering = ordering.copy()
        
        t = new_ordering[u]
        new_ordering[u] = new_ordering[v]
        new_ordering[v] = t
        
        return new_ordering
    
    # Select flip function based on method
    if flip_method == "flip2":
        flip_func = flip2
    elif flip_method == "flip3":
        flip_func = flip3
    elif flip_method == "flipEdge":
        flip_func = flipEdge
    else:
        raise ValueError(f"Unknown flip method: {flip_method}")
    
    while iter_count < iter_max:
        iter_count += 1
        
        new_ordering = flip_func()
        
        current_cost = calculate_min_linear_arrangement(G, ordering)
        neighbor_cost = calculate_min_linear_arrangement(G, new_ordering)
        
        if current_cost > neighbor_cost:
            ordering = new_ordering
            
    return calculate_min_linear_arrangement(G, ordering)
        
if __name__ == "__main__":
    # Read file from Dataset/processed and create graph
    # Then calculate spectral sequencing cost and store in csv file 

    random.seed(42)
    # Get processed datasets directory
    dataset_dir = Path(__file__).parent.parent / "Dataset" / "processed"
    results_dir = Path(__file__).parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Output CSV file
    output_file = results_dir / "local_search_results.csv"
    
    # Find all preprocessed graph files
    graph_files = sorted(dataset_dir.glob("*_preprocessed.txt"))
    
    if not graph_files:
        print(f"No preprocessed graph files found in {dataset_dir}")
        exit(1)
    
    print(f"Processing {len(graph_files)} graph files...")
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
            
            # Test all three flip methods
            for method in ["flip2", "flip3", "flipEdge"]:
                initial_ordering = list(G.nodes())
                random.shuffle(initial_ordering)
                cost = local_search(G, initial_ordering, flip_method=method)
                
                # Store result
                results.append({
                    'dataset': graph_file.stem.replace('_preprocessed', ''),
                    'vertices': n,
                    'edges': m,
                    'method': method,
                    'local_search_cost': cost
                })
                
                print(f"  {method:10s} - Cost: {cost}")
            
        except Exception as e:
            print(f"  Error processing {graph_file.name}: {e}")
            continue
    
    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'vertices', 'edges', 'method', 'local_search_cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(results)} datasets successfully")
    else:
        print("\nNo results to save")

    
        