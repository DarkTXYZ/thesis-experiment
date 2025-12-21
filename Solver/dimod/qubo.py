import dimod
from dwave.samplers import PathIntegralAnnealingSampler, SteepestDescentSampler
from pyqubo import Array, Binary
import networkx as nx
import csv
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Directory paths
    'dataset_dir': Path(__file__).parent.parent.parent / "Dataset" / "small_graph",
    'results_dir': Path(__file__).parent.parent.parent / "Results",
    
    # QUBO parameters
    'lambda_multiplier': 1.0,  # Multiplier for lambda (lambda = N * multiplier)
    
    # Sampling parameters
    'num_reads': 5000,
    'num_sweeps': 50,
    
    # Output settings
    'output_prefix': 'dimod_results',
}

# ============================================================================
# QUBO GENERATION
# ============================================================================

def generate_minla_qubo(G : nx.Graph):
    
    N = G.number_of_nodes()
    
    _lambda = N * CONFIG['lambda_multiplier']
    
    
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

# ============================================================================
# SOLUTION DECODING
# ============================================================================

def decode_solution(solution_dict, N):
    """Decode binary solution to ordering and check feasibility
    
    Returns:
        tuple: (ordering, is_feasible) where ordering is the position assignment
               and is_feasible is True if all constraints are satisfied
    """
    is_feasible = True
    
    # Check thermometer constraint
    for u in range(N):
        for k in range(N-1):
            if solution_dict.get(f'X[{u}][{k}]', 0) == 0 and solution_dict.get(f'X[{u}][{k+1}]', 0) == 1:
                is_feasible = False
                break
        if not is_feasible:
            break
    
    # Check bijective constraint
    cnt_label = set()
    for u in range(N):
        label = sum(solution_dict.get(f'X[{u}][{k}]', 0) for k in range(N))
        cnt_label.add(label)
        
    if len(cnt_label) != N or cnt_label != set(range(1, N+1)):
        is_feasible = False
    
    # Decode ordering
    ordering = [0] * N
    for u in range(N):
        position = 0
        for k in range(N):
            position += solution_dict.get(f'X[{u}][{k}]', 0)
        ordering[u] = position
    
    return ordering, is_feasible

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Setup paths from configuration
    processed_dir = CONFIG['dataset_dir']
    results_dir = CONFIG['results_dir']
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get all preprocessed graph files
    graph_files = sorted(processed_dir.glob("*_preprocessed.txt"))
    
    # Define samplers to test
    samplers = [
        ('PathIntegralAnnealingSampler', PathIntegralAnnealingSampler()),
        ('SteepestDescentSampler', SteepestDescentSampler()),
    ]
    
    print("=" * 80)
    print("DIMOD Samplers for MINLA QUBO")
    print("=" * 80)
    
    results = []
    
    for sampler_name, sampler in samplers:
        print(f"\n{'=' * 80}")
        print(f"Testing: {sampler_name}")
        print(f"{'=' * 80}")
        
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
                
                # Solve using current sampler
                print(f"  Solving with {sampler_name}...")
                sampleset = sampler.sample(bqm, num_reads=CONFIG['num_reads'], num_sweeps=CONFIG['num_sweeps'])
                
                # Get best solution
                solution = sampleset.lowest().to_pandas_dataframe()
                energy = solution['energy'].values[0]
                
                # Decode solution to ordering and check feasibility
                solution_dict = solution.iloc[0].to_dict()
                ordering, is_feasible = decode_solution(solution_dict, n)
                
                # Calculate actual cost
                cost = calculate_min_linear_arrangement(G, ordering)
                
                # Store result
                results.append({
                    'dataset': graph_file.stem.replace('_preprocessed', ''),
                    'vertices': n,
                    'edges': m,
                    'sampler': sampler_name,
                    'num_reads': CONFIG['num_reads'],
                    'energy': energy,
                    'minla_cost': cost,
                    'feasible': is_feasible
                })
                
                print(f"  Energy: {energy}, MINLA Cost: {cost}, Feasible: {is_feasible}")
                
            except Exception as e:
                print(f"  Error processing {graph_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Write results to CSV
    if results:
        output_file = results_dir / f"{CONFIG['output_prefix']}_{timestamp}.csv"
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'vertices', 'edges', 'sampler', 'num_reads', 'energy', 'minla_cost', 'feasible']
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