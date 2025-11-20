import networkx as nx
from typing import List
import sys
from pathlib import Path
import csv
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.append(str(Path(__file__).parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def spectral_sequencing(graph: nx.Graph) -> List[int]:
    return nx.linalg.spectral_ordering(graph, weight=None ,method='tracemin_lu')

def calculate_spectral_sequencing_cost(graph: nx.Graph) -> int:
    """Calculate the cost of the linear arrangement using spectral ordering."""
    ordering = spectral_sequencing(graph)
    cost = calculate_min_linear_arrangement(graph, ordering)
    return cost

if __name__ == "__main__":
    # Read file from Dataset/processed and create graph
    # Then calculate spectral sequencing cost and store in csv file 

    
    # Get processed datasets directory
    dataset_dir = Path(__file__).parent.parent / "Dataset" / "processed"
    results_dir = Path(__file__).parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Output CSV file
    output_file = results_dir / "spectral_sequencing_results.csv"
    
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
            
            # Calculate spectral sequencing cost
            cost = calculate_spectral_sequencing_cost(G)
            
            # Store result
            results.append({
                'dataset': graph_file.stem.replace('_preprocessed', ''),
                'vertices': n,
                'edges': m,
                'spectral_cost': cost
            })
            
            print(f"  Vertices: {n}, Edges: {m}")
            print(f"  Spectral Sequencing Cost: {cost}")
            
        except Exception as e:
            print(f"  Error processing {graph_file.name}: {e}")
            continue
    
    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'vertices', 'edges', 'spectral_cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(results)} datasets successfully")
    else:
        print("\nNo results to save")
