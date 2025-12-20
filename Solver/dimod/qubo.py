import dimod
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
from pyqubo import Array, Binary
import networkx as nx
import csv
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement

def generate_minla_qubo(G : nx.Graph):
    
    N = G.number_of_nodes()
    
    _lambda = N
    
    
    X = Array.create('X', shape=(N, N), vartype='BINARY')
    
    H_thermometer = 0
    for u in range(N):
        for k in range(N-1):
            H_thermometer += _lambda * (1 - X[u][k]) * X[u][k+1]
            
    H_bijective = 0
    for k in range(N):
        cnt = 0
        for u in range(N):
            cnt += X[u][k]
        H_bijective += _lambda * ((N - k) - cnt) * ((N - k) - cnt)
        
    H_objective = 0
    for u,v in G.edges:
        diff = 0
        for k in range(N):
            diff += X[u][k] + X[v][k] - 2 * X[u][k] * X[v][k]
        H_objective += diff
    
    H = (H_thermometer + H_bijective) + H_objective
    model = H.compile()
    return model.to_bqm()

def decode_solution(solution_dict, N):
    """Decode binary solution to ordering"""
    
    for u in range(N):
        for k in range(N-1):
            if solution_dict.get(f'X[{u}][{k}]', 0) == 0 and solution_dict.get(f'X[{u}][{k+1}]', 0) == 1:
                raise ValueError("Invalid solution: Thermometer constraint violated")
    
    ordering = [0] * N
    for u in range(N):
        position = 0
        for k in range(N):
            position += solution_dict.get(f'X[{u}][{k}]', 0)
        ordering[u] = position
    return ordering

def main():
    # Setup paths
    processed_dir = Path(__file__).parent.parent.parent / "Dataset" / "processed"
    results_dir = Path(__file__).parent.parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"dimod_pathintegral_results_{timestamp}.csv"
    
    # Get all preprocessed graph files
    graph_files = sorted(processed_dir.glob("*_preprocessed.txt"))
    
    print("=" * 80)
    print("DIMOD PathIntegralAnnealingSampler for MINLA QUBO")
    print("=" * 80)
    
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
            
            # Generate QUBO
            print(f"  Generating QUBO...")
            bqm = generate_minla_qubo(G)
            
            # Solve using PathIntegralAnnealingSampler
            print(f"  Solving with PathIntegralAnnealingSampler...")
            sampler = PathIntegralAnnealingSampler()
            sampleset = sampler.sample(bqm)
            
            # Get best solution
            solution = sampleset.lowest().to_pandas_dataframe()
            energy = solution['energy'].values[0]
            
            # Decode solution to ordering
            solution_dict = solution.iloc[0].to_dict()
            ordering = decode_solution(solution_dict, n)
            
            # Calculate actual cost
            cost = calculate_min_linear_arrangement(G, ordering)
            
            # Store result
            results.append({
                'dataset': graph_file.stem.replace('_preprocessed', ''),
                'vertices': n,
                'edges': m,
                'sampler': 'PathIntegralAnnealingSampler',
                'num_reads': 1000,
                'energy': energy,
                'minla_cost': cost
            })
            
            print(f"  Energy: {energy}, MINLA Cost: {cost}")
            
        except Exception as e:
            print(f"  Error processing {graph_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'vertices', 'edges', 'sampler', 'num_reads', 'energy', 'minla_cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(results)} datasets successfully")
    else:
        print("\nNo results to save")

if __name__ == "__main__":
    main()