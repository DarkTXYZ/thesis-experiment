import os
import csv
import time
import pickle
import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
from Solver.HeuristicSolver.OpenJijSolver import OpenJijSolver
from Solver.HeuristicSolver.QWSamplerSolver import QWaveSamplerSolver
from Solver.HeuristicSolver.SBSolver import SimulatedBifurcationSolver
from Solver.penalty_coefficients import calculate_exact_bound, calculate_lucas_bound
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for the quantum experiment."""
    
    # Dataset settings
    vertex_counts: list = field(default_factory=lambda: [5, 8, 11, 13, 15])
    dataset_dir: str = "Dataset/quantum_dataset"
    
    # Solver settings
    penalty_methods: list = field(default_factory=lambda: ['exact', 'lucas'])
    num_reads: int = 1000
    seed: int = 42
    
    # Solvers to use (set to False to disable)
    use_openjij: bool = True
    use_qwavesampler: bool = True
    use_simulated_bifurcation: bool = True
    
    # Success criteria
    success_gap_threshold: float = 0.0  # Success if relative gap <= this value (0.0 = exact match, 0.1 = 10% gap)
    
    # Output settings
    results_dir: str = "Results"
    save_detailed: bool = True
    save_aggregated: bool = True
    
    # Logging
    verbose: bool = True


# Default configuration - MODIFY THIS TO CHANGE EXPERIMENT PARAMETERS
CONFIG = ExperimentConfig(
    vertex_counts=[5, 8, 11, 13, 15],
    penalty_methods=['exact', 'lucas'],
    num_reads=100,
    seed=42,
    use_openjij=True,
    use_qwavesampler=False,
    use_simulated_bifurcation=False,
    success_gap_threshold=0.05,  # 5% gap allowed for success
    verbose=True
)
# =============================================================================


def load_dataset(filepath: str) -> dict:
    """Load a dataset from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def build_graph_from_edges(edges: list, num_vertices: int) -> nx.Graph:
    """Build a NetworkX graph from edge list."""
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    G.add_edges_from(edges)
    return G


def calculate_metrics(results: list, best_costs: list, success_gap_threshold: float = 0.0) -> dict:
    """
    Calculate experiment metrics.
    
    Args:
        results: List of (minla_cost, is_feasible) tuples
        best_costs: List of best known costs from baseline
        success_gap_threshold: Maximum relative gap to count as success (0.0 = exact match, 0.1 = 10% gap)
    
    Returns:
        Dictionary with feasibility_rate, success_rate, dominance_score, avg_relative_gap
    """
    num_graphs = len(results)
    
    # Feasibility rate
    num_feasible = sum(1 for _, is_feasible in results if is_feasible)
    feasibility_rate = num_feasible / num_graphs
    
    # Success rate (feasible and relative gap <= threshold)
    num_success = 0
    for (minla_cost, is_feasible), best_cost in zip(results, best_costs):
        if is_feasible and minla_cost is not None and best_cost > 0:
            relative_gap = (minla_cost - best_cost) / best_cost
            if relative_gap <= success_gap_threshold:
                num_success += 1
    success_rate = num_success / num_graphs
    
    # Dominance score (how many times solver beats or matches baseline, among feasible)
    num_dominant = sum(
        1 for (minla_cost, is_feasible), best_cost in zip(results, best_costs)
        if is_feasible and minla_cost is not None and minla_cost <= best_cost
    )
    dominance_score = num_dominant / num_feasible if num_feasible > 0 else 0.0
    
    # Average relative gap: (solver_cost - best_cost) / best_cost, only for feasible solutions
    relative_gaps = []
    for (minla_cost, is_feasible), best_cost in zip(results, best_costs):
        if is_feasible and minla_cost is not None and best_cost > 0:
            gap = (minla_cost - best_cost) / best_cost
            relative_gaps.append(gap)
    avg_relative_gap = sum(relative_gaps) / len(relative_gaps) if relative_gaps else float('inf')
    
    return {
        'feasibility_rate': feasibility_rate,
        'success_rate': success_rate,
        'dominance_score': dominance_score,
        'avg_relative_gap': avg_relative_gap,
        'num_feasible': num_feasible,
        'num_success': num_success
    }


def run_experiment(config: ExperimentConfig = None):
    """Run quantum experiment on all datasets."""
    if config is None:
        config = CONFIG
    
    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, config.dataset_dir)
    results_dir = os.path.join(base_dir, config.results_dir)
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize solvers based on configuration
    solvers = {}
    if config.use_openjij:
        solvers['OpenJij'] = OpenJijSolver()
    if config.use_qwavesampler:
        solvers['QWaveSampler'] = QWaveSamplerSolver()
    if config.use_simulated_bifurcation:
        solvers['SimulatedBifurcation'] = SimulatedBifurcationSolver()
    
    if not solvers:
        print("No solvers enabled. Please enable at least one solver in config.")
        return [], []
    
    # Configure all solvers with common parameters
    for solver in solvers.values():
        solver.configure(seed=config.seed, num_reads=config.num_reads)
    
    if config.verbose:
        print("=" * 60)
        print("QUANTUM EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"Vertex counts: {config.vertex_counts}")
        print(f"Penalty methods: {config.penalty_methods}")
        print(f"Num reads: {config.num_reads}")
        print(f"Seed: {config.seed}")
        print(f"Solvers: {list(solvers.keys())}")
        print("=" * 60)
    
    # Store aggregated results
    aggregated_results = []
    detailed_results = []
    
    # Calculate total iterations for progress bar
    total_configs = len(config.vertex_counts) * len(config.penalty_methods) * len(solvers)
    pbar_configs = tqdm(total=total_configs, desc="Configurations", position=0)
    
    for n in config.vertex_counts:
        # Load dataset
        dataset_path = os.path.join(dataset_dir, f'quantum_n{n}.pkl')
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            continue
            
        dataset = load_dataset(dataset_path)
        num_graphs = dataset['num_graphs']
        density = dataset['density']
        graphs_data = dataset['graphs']
        
        if config.verbose:
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Testing dataset: n={n}, {num_graphs} graphs, density={density}")
            tqdm.write(f"{'='*60}")
        
        for penalty_mode in config.penalty_methods:
            if config.verbose:
                tqdm.write(f"\n  Penalty mode: {penalty_mode}")
            
            for solver_name, solver in solvers.items():
                pbar_configs.set_description(f"n={n}, {penalty_mode}, {solver_name}")
                
                if config.verbose:
                    tqdm.write(f"    Solver: {solver_name}...")
                
                # Configure solver with penalty mode
                solver.configure(penalty_mode=penalty_mode)
                
                # Collect results for all graphs in this dataset
                solver_results = []
                best_costs = []
                total_time = 0
                
                for graph_data in tqdm(graphs_data, desc=f"    Graphs", leave=False, position=1):
                    graph_id = graph_data['id']
                    edges = graph_data['edges']
                    best_cost = graph_data['best_cost']
                    best_costs.append(best_cost)
                    
                    # Build graph
                    graph = build_graph_from_edges(edges, n)
                    m = graph.number_of_edges()
                    
                    # Get penalty values
                    if penalty_mode == 'exact':
                        mu_thermo, mu_bijec = calculate_exact_bound(graph)
                    else:  # lucas
                        mu_thermo, mu_bijec = calculate_lucas_bound(graph)
                    
                    # Solve and measure time
                    start_time = time.time()
                    result = solver.solve(graph)
                    solve_time = time.time() - start_time
                    total_time += solve_time
                    
                    # Calculate MinLA cost only if solution is feasible
                    if result.is_feasible:
                        minla_cost = calculate_min_linear_arrangement(graph, result.ordering)
                    else:
                        minla_cost = None
                    
                    solver_results.append((minla_cost, result.is_feasible))
                    
                    # Store detailed result
                    detailed_results.append({
                        'solver_name': solver_name,
                        'num_vertices': n,
                        'graph_id': graph_id,
                        'num_edges': m,
                        'penalty_mode': penalty_mode,
                        'mu_thermometer': mu_thermo,
                        'mu_bijective': mu_bijec,
                        'energy': result.energy,
                        'minla_cost': minla_cost,
                        'best_known_cost': best_cost,
                        'is_feasible': result.is_feasible,
                        'solve_time': solve_time
                    })
                
                # Calculate metrics
                metrics = calculate_metrics(solver_results, best_costs, config.success_gap_threshold)
                
                # Store aggregated result
                aggregated_results.append({
                    'solver_name': solver_name,
                    'num_vertices': n,
                    'num_graphs': num_graphs,
                    'density': density,
                    'penalty_mode': penalty_mode,
                    'feasibility_rate': metrics['feasibility_rate'],
                    'success_rate': metrics['success_rate'],
                    'dominance_score': metrics['dominance_score'],
                    'avg_relative_gap': metrics['avg_relative_gap'],
                    'num_feasible': metrics['num_feasible'],
                    'num_success': metrics['num_success'],
                    'total_time': total_time
                })
                
                pbar_configs.update(1)
                
                if config.verbose:
                    tqdm.write(f"      Feasibility: {metrics['feasibility_rate']*100:.1f}% ({metrics['num_feasible']}/{num_graphs})")
                    tqdm.write(f"      Success rate: {metrics['success_rate']*100:.1f}% ({metrics['num_success']}/{num_graphs})")
                    tqdm.write(f"      Dominance: {metrics['dominance_score']*100:.1f}%")
                    tqdm.write(f"      Avg relative gap: {metrics['avg_relative_gap']*100:.2f}%")
                    tqdm.write(f"      Total time: {total_time:.2f}s")
    
    pbar_configs.close()
    
    # Save aggregated results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config.save_aggregated:
        agg_output_file = os.path.join(results_dir, f'quantum_experiment_aggregated_{timestamp}.csv')
        agg_fieldnames = ['solver_name', 'num_vertices', 'num_graphs', 'density', 'penalty_mode',
                          'feasibility_rate', 'success_rate', 'dominance_score', 'avg_relative_gap',
                          'num_feasible', 'num_success', 'total_time']
        
        with open(agg_output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_results)
        
        if config.verbose:
            print(f"\nAggregated results saved to: {agg_output_file}")
    
    # Save detailed results to CSV
    if config.save_detailed:
        detail_output_file = os.path.join(results_dir, f'quantum_experiment_detailed_{timestamp}.csv')
        detail_fieldnames = ['solver_name', 'num_vertices', 'graph_id', 'num_edges', 'penalty_mode',
                             'mu_thermometer', 'mu_bijective', 'energy', 'minla_cost', 
                             'best_known_cost', 'is_feasible', 'solve_time']
        
        with open(detail_output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
        
        if config.verbose:
            print(f"Detailed results saved to: {detail_output_file}")
    
    return aggregated_results, detailed_results


if __name__ == "__main__":
    run_experiment()
