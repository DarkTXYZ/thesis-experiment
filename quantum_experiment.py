import os
import csv
import time
import networkx as nx
from datetime import datetime
from Solver.HeuristicSolver.OpenJijSolver import OpenJijSolver
from Solver.HeuristicSolver.QWSamplerSolver import QWaveSamplerSolver
from Solver.HeuristicSolver.SBSolver import SimulatedBifurcationSolver
from Solver.penalty_coefficients import calculate_exact_bound, calculate_lucas_bound
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def load_graph(filepath: str) -> nx.Graph:
    """Load a graph from .edgelist file."""
    return nx.read_edgelist(filepath, nodetype=int)


def calculate_constraint_values(raw_sample: dict, n: int, mu_thermo: float, mu_bijec: float):
    """Calculate H_thermometer and H_bijective constraint values from raw sample."""
    # Build X matrix from raw sample
    X = [[raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)] for u in range(n)]
    
    # H_thermometer: sum of (1 - X[u][k]) * X[u][k+1] violations
    h_thermo_violations = sum(
        (1 - X[u][k]) * X[u][k+1]
        for u in range(n)
        for k in range(n-1)
    )
    H_thermometer = mu_thermo * h_thermo_violations
    
    # H_bijective: sum of ((n - k) - sum(X[u][k]))^2
    h_bijec_violations = sum(
        ((n - k) - sum(X[u][k] for u in range(n))) ** 2
        for k in range(n)
    )
    H_bijective = mu_bijec * h_bijec_violations
    
    return H_thermometer, h_thermo_violations, H_bijective, h_bijec_violations


def get_all_graphs(dataset_dir: str) -> list:
    """Get all graphs from quantum_dataset directory."""
    graphs = []
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.edgelist'):
            filepath = os.path.join(dataset_dir, filename)
            graph = load_graph(filepath)
            graph_name = filename.replace('.edgelist', '')
            graphs.append((graph_name, graph))
    
    return sorted(graphs, key=lambda x: x[0])


def run_experiment():
    """Run OpenJijSolver experiment with exact and lucas penalty methods."""
    dataset_dir = os.path.join(os.path.dirname(__file__), 'Dataset', 'quantum_dataset')
    results_dir = os.path.join(os.path.dirname(__file__), 'Results')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all graphs from quantum_dataset
    graphs = get_all_graphs(dataset_dir)
    print(f"Found {len(graphs)} graphs in quantum_dataset")
    
    # Penalty methods to test
    penalty_methods = ['exact', 'lucas']
    
    # Number of reads for the sampler
    num_reads = 1000
    seed = 42
    
    # Prepare results
    results = []
    
    # Initialize all 3 solvers
    solvers = {
        'OpenJij': OpenJijSolver(),
        'QWaveSampler': QWaveSamplerSolver(),
        'SimulatedBifurcation': SimulatedBifurcationSolver()
    }
    
    # Configure all solvers with common parameters
    for solver in solvers.values():
        solver.configure(seed=seed, num_reads=num_reads)
    
    for graph_name, graph in graphs:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        print(f"\nProcessing: {graph_name} (n={n}, m={m})")
        
        for penalty_mode in penalty_methods:
            print(f"  Testing with {penalty_mode} penalty...")
            
            # Get penalty values
            if penalty_mode == 'exact':
                mu_thermo, mu_bijec = calculate_exact_bound(graph)
            else:  # lucas
                mu_thermo, mu_bijec = calculate_lucas_bound(graph)
            
            for solver_name, solver in solvers.items():
                print(f"    Solver: {solver_name}...")
                
                # Configure solver
                solver.configure(penalty_mode=penalty_mode)
                
                # Solve and measure time
                start_time = time.time()
                result = solver.solve(graph)
                solve_time = time.time() - start_time
                
                # Calculate MinLA cost only if solution is feasible
                if result.is_feasible:
                    minla_cost = calculate_min_linear_arrangement(graph, result.ordering)
                else:
                    minla_cost = None  # Invalid ordering
                
                # Store result
                results.append({
                    'solver_name': solver_name,
                    'graph_name': graph_name,
                    'num_vertices': n,
                    'num_edges': m,
                    'penalty_mode': penalty_mode,
                    'mu_thermometer': mu_thermo,
                    'mu_bijective': mu_bijec,
                    'energy': result.energy,
                    'minla_cost': minla_cost,
                    'is_feasible': result.is_feasible
                })
                
                print(f"      Energy: {result.energy}, MinLA: {minla_cost}, Feasible: {result.is_feasible}, Time: {solve_time:.4f}s")
                if not result.is_feasible and hasattr(result, 'raw_sample') and result.raw_sample:
                    # Print QUBO variable assignments as matrix X[u][k]
                    print(f"      QUBO assignment (X[u][k] matrix):")
                    for u in range(n):
                        row = [result.raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
                        print(f"        u={u}: {row}")
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'quantum_experiment_reads{num_reads}_{timestamp}.csv')
    
    fieldnames = ['solver_name', 'graph_name', 'num_vertices', 'num_edges', 'penalty_mode', 
                  'mu_thermometer', 'mu_bijective', 'energy', 
                  'minla_cost', 'is_feasible']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    run_experiment()
