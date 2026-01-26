import os
import csv
import time
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import networkx as nx
from tqdm import tqdm

from Solver.HeuristicSolver.OpenJijSolver import OpenJijSolver
from Solver.HeuristicSolver.QWSamplerSolver import QWaveSamplerSolver
from Solver.HeuristicSolver.SBSolver import SimulatedBifurcationSolver
from Solver.penalty_coefficients import calculate_exact_bound, calculate_lucas_bound
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for the quantum experiment."""
    
    # Dataset settings
    vertex_counts: list[int] = field(default_factory=lambda: [5, 8, 11, 13, 15])
    dataset_dir: str = "Dataset/quantum_dataset"
    
    # Solver settings
    penalty_methods: list[str] = field(default_factory=lambda: ['exact', 'lucas'])
    num_reads: int = 1000
    seed: int = 42
    
    # Solvers to use (set to False to disable)
    use_openjij: bool = True
    use_qwavesampler: bool = True
    use_simulated_bifurcation: bool = True
    
    # Success criteria
    success_gap_threshold: float = 0.0  # 0.0 = exact match, 0.1 = 10% gap allowed
    
    # Output settings
    results_dir: str = "Results"
    save_detailed: bool = True
    save_aggregated: bool = True
    
    # Logging
    verbose: bool = True


@dataclass
class GraphResult:
    """Result for a single graph."""
    minla_cost: Optional[int]
    is_feasible: bool


@dataclass
class DetailedResult:
    """Detailed result for a single graph solve."""
    solver_name: str
    num_vertices: int
    graph_id: int
    num_edges: int
    penalty_mode: str
    mu_thermometer: float
    mu_bijective: float
    energy: float
    minla_cost: Optional[int]
    best_known_cost: int
    is_feasible: bool
    solve_time: float


@dataclass
class AggregatedResult:
    """Aggregated result for a dataset configuration."""
    solver_name: str
    num_vertices: int
    num_graphs: int
    density: float
    penalty_mode: str
    feasibility_rate: float
    success_rate: float
    dominance_score: float
    avg_relative_gap: float
    num_feasible: int
    num_success: int
    total_time: float


@dataclass
class Metrics:
    """Calculated metrics for experiment results."""
    feasibility_rate: float
    success_rate: float
    dominance_score: float
    avg_relative_gap: float
    num_feasible: int
    num_success: int


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
CONFIG = ExperimentConfig(
    vertex_counts=[5, 8, 11, 13, 15],
    penalty_methods=['exact', 'lucas'],
    num_reads=100,
    seed=42,
    use_openjij=True,
    use_qwavesampler=False,
    use_simulated_bifurcation=False,
    success_gap_threshold=0.1,
    verbose=True
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_dataset(filepath: str) -> dict:
    """Load a dataset from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def build_graph(edges: list[tuple[int, int]], num_vertices: int) -> nx.Graph:
    """Build a NetworkX graph from edge list."""
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    G.add_edges_from(edges)
    return G


def get_penalty_bounds(graph: nx.Graph, penalty_mode: str) -> tuple[float, float]:
    """Get penalty bounds based on the penalty mode."""
    if penalty_mode == 'exact':
        return calculate_exact_bound(graph)
    return calculate_lucas_bound(graph)


def calculate_relative_gap(solver_cost: int, best_cost: int) -> Optional[float]:
    """Calculate relative gap between solver cost and best known cost."""
    if best_cost <= 0:
        return None
    return (solver_cost - best_cost) / best_cost


def calculate_metrics(
    results: list[GraphResult], 
    best_costs: list[int], 
    success_gap_threshold: float = 0.0
) -> Metrics:
    """
    Calculate experiment metrics.
    
    Args:
        results: List of GraphResult objects
        best_costs: List of best known costs from baseline
        success_gap_threshold: Maximum relative gap to count as success
    
    Returns:
        Metrics dataclass with calculated values
    """
    num_graphs = len(results)
    
    # Count feasible and successful solutions
    num_feasible = 0
    num_success = 0
    num_dominant = 0
    relative_gaps: list[float] = []
    
    for result, best_cost in zip(results, best_costs):
        if not result.is_feasible or result.minla_cost is None:
            continue
            
        num_feasible += 1
        
        if result.minla_cost <= best_cost:
            num_dominant += 1
        
        gap = calculate_relative_gap(result.minla_cost, best_cost)
        if gap is not None:
            relative_gaps.append(gap)
            if gap <= success_gap_threshold:
                num_success += 1
    
    return Metrics(
        feasibility_rate=num_feasible / num_graphs,
        success_rate=num_success / num_graphs,
        dominance_score=num_dominant / num_feasible if num_feasible > 0 else 0.0,
        avg_relative_gap=sum(relative_gaps) / len(relative_gaps) if relative_gaps else float('inf'),
        num_feasible=num_feasible,
        num_success=num_success
    )


def init_solvers(config: ExperimentConfig) -> dict[str, object]:
    """Initialize solvers based on configuration."""
    solvers = {}
    if config.use_openjij:
        solvers['OpenJij'] = OpenJijSolver()
    if config.use_qwavesampler:
        solvers['QWaveSampler'] = QWaveSamplerSolver()
    if config.use_simulated_bifurcation:
        solvers['SimulatedBifurcation'] = SimulatedBifurcationSolver()
    
    for solver in solvers.values():
        solver.configure(seed=config.seed, num_reads=config.num_reads)
    
    return solvers


def process_single_graph(
    graph_data: dict,
    num_vertices: int,
    solver: object,
    penalty_mode: str,
    solver_name: str
) -> tuple[GraphResult, DetailedResult, float]:
    """
    Process a single graph with the given solver.
    
    Returns:
        Tuple of (GraphResult, DetailedResult, solve_time)
    """
    graph_id = graph_data['id']
    edges = graph_data['edges']
    best_cost = graph_data['best_cost']
    
    graph = build_graph(edges, num_vertices)
    num_edges = graph.number_of_edges()
    
    mu_thermo, mu_bijec = get_penalty_bounds(graph, penalty_mode)
    
    start_time = time.time()
    result = solver.solve(graph)
    solve_time = time.time() - start_time
    
    minla_cost = None
    if result.is_feasible:
        minla_cost = calculate_min_linear_arrangement(graph, result.ordering)
    
    graph_result = GraphResult(minla_cost=minla_cost, is_feasible=result.is_feasible)
    
    detailed = DetailedResult(
        solver_name=solver_name,
        num_vertices=num_vertices,
        graph_id=graph_id,
        num_edges=num_edges,
        penalty_mode=penalty_mode,
        mu_thermometer=mu_thermo,
        mu_bijective=mu_bijec,
        energy=result.energy,
        minla_cost=minla_cost,
        best_known_cost=best_cost,
        is_feasible=result.is_feasible,
        solve_time=solve_time
    )
    
    return graph_result, detailed, best_cost


def log_config(config: ExperimentConfig, solvers: dict[str, object]) -> None:
    """Log experiment configuration."""
    if not config.verbose:
        return
    print("=" * 60)
    print("QUANTUM EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Vertex counts: {config.vertex_counts}")
    print(f"Penalty methods: {config.penalty_methods}")
    print(f"Num reads: {config.num_reads}")
    print(f"Seed: {config.seed}")
    print(f"Success gap threshold: {config.success_gap_threshold * 100:.1f}%")
    print(f"Solvers: {list(solvers.keys())}")
    print("=" * 60)


def log_metrics(metrics: Metrics, num_graphs: int, total_time: float) -> None:
    """Log metrics for a solver run."""
    tqdm.write(f"      Feasibility: {metrics.feasibility_rate*100:.1f}% ({metrics.num_feasible}/{num_graphs})")
    tqdm.write(f"      Success rate: {metrics.success_rate*100:.1f}% ({metrics.num_success}/{num_graphs})")
    tqdm.write(f"      Dominance: {metrics.dominance_score*100:.1f}%")
    tqdm.write(f"      Avg relative gap: {metrics.avg_relative_gap*100:.2f}%")
    tqdm.write(f"      Total time: {total_time:.2f}s")


def save_results_to_csv(
    results: list,
    filepath: str,
    fieldnames: list[str],
    is_dataclass: bool = True
) -> None:
    """Save results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if is_dataclass:
            writer.writerows([vars(r) for r in results])
        else:
            writer.writerows(results)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment(config: ExperimentConfig = None) -> tuple[list[AggregatedResult], list[DetailedResult]]:
    """Run quantum experiment on all datasets."""
    if config is None:
        config = CONFIG
    
    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, config.dataset_dir)
    results_dir = os.path.join(base_dir, config.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    solvers = init_solvers(config)
    if not solvers:
        print("No solvers enabled. Please enable at least one solver in config.")
        return [], []
    
    log_config(config, solvers)
    
    aggregated_results: list[AggregatedResult] = []
    detailed_results: list[DetailedResult] = []
    
    total_configs = len(config.vertex_counts) * len(config.penalty_methods) * len(solvers)
    pbar_configs = tqdm(total=total_configs, desc="Configurations", position=0)
    
    for n in config.vertex_counts:
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
                
                solver.configure(penalty_mode=penalty_mode)
                
                graph_results: list[GraphResult] = []
                best_costs: list[int] = []
                total_time = 0.0
                
                for graph_data in tqdm(graphs_data, desc="    Graphs", leave=False, position=1):
                    result, detailed, best_cost = process_single_graph(
                        graph_data, n, solver, penalty_mode, solver_name
                    )
                    graph_results.append(result)
                    best_costs.append(best_cost)
                    total_time += detailed.solve_time
                    detailed_results.append(detailed)
                
                metrics = calculate_metrics(graph_results, best_costs, config.success_gap_threshold)
                
                aggregated_results.append(AggregatedResult(
                    solver_name=solver_name,
                    num_vertices=n,
                    num_graphs=num_graphs,
                    density=density,
                    penalty_mode=penalty_mode,
                    feasibility_rate=metrics.feasibility_rate,
                    success_rate=metrics.success_rate,
                    dominance_score=metrics.dominance_score,
                    avg_relative_gap=metrics.avg_relative_gap,
                    num_feasible=metrics.num_feasible,
                    num_success=metrics.num_success,
                    total_time=total_time
                ))
                
                pbar_configs.update(1)
                
                if config.verbose:
                    log_metrics(metrics, num_graphs, total_time)
    
    pbar_configs.close()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config.save_aggregated and aggregated_results:
        agg_path = os.path.join(results_dir, f'quantum_experiment_aggregated_{timestamp}.csv')
        agg_fields = [
            'solver_name', 'num_vertices', 'num_graphs', 'density', 'penalty_mode',
            'feasibility_rate', 'success_rate', 'dominance_score', 'avg_relative_gap',
            'num_feasible', 'num_success', 'total_time'
        ]
        save_results_to_csv(aggregated_results, agg_path, agg_fields)
        if config.verbose:
            print(f"\nAggregated results saved to: {agg_path}")
    
    if config.save_detailed and detailed_results:
        detail_path = os.path.join(results_dir, f'quantum_experiment_detailed_{timestamp}.csv')
        detail_fields = [
            'solver_name', 'num_vertices', 'graph_id', 'num_edges', 'penalty_mode',
            'mu_thermometer', 'mu_bijective', 'energy', 'minla_cost',
            'best_known_cost', 'is_feasible', 'solve_time'
        ]
        save_results_to_csv(detailed_results, detail_path, detail_fields)
        if config.verbose:
            print(f"Detailed results saved to: {detail_path}")
    
    return aggregated_results, detailed_results


if __name__ == "__main__":
    run_experiment()
