import os
import csv
import time
import pickle
import statistics
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
    use_synthetic_dataset: bool = True
    
    # Real-world dataset settings
    use_real_world_dataset: bool = False
    real_world_dataset_path: str = "Dataset/quantum_real_world_dataset/quantum_real_world.pkl"
    
    # Solver settings
    penalty_methods: list[str] = field(default_factory=lambda: ['exact', 'lucas'])
    num_reads: int = 1000
    seed: int = 42
    
    # Solvers to use (set to False to disable)
    use_openjij: bool = True
    use_qwavesampler: bool = True
    use_simulated_bifurcation: bool = True
    
    # QWaveSampler settings
    qwavesampler_types: list[str] = field(default_factory=lambda: ['path'])  # Options: 'path', 'sa', 'steepest'
    
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
    sampler_type: str
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
    """Aggregated result for a dataset configuration (synthetic only)."""
    solver_name: str
    sampler_type: str
    dataset_name: str
    num_vertices: int
    num_graphs: int
    density: float
    penalty_mode: str
    feasibility_rate: float
    success_rate: float
    dominance_score: float
    avg_relative_gap: float
    std_relative_gap: float
    num_feasible: int
    num_success: int
    total_time: float


@dataclass
class RealWorldResult:
    """Individual result for a real-world graph (not aggregated)."""
    solver_name: str
    sampler_type: str
    graph_name: str
    num_vertices: int
    num_edges: int
    penalty_mode: str
    mu_thermometer: float
    mu_bijective: float
    is_feasible: bool
    objective_value: Optional[int]  # MINLA cost from solver
    spectral_cost: int
    successive_augmentation_cost: int
    local_search_cost: int
    best_known_cost: int
    relative_gap: Optional[float]  # Gap from best_known_cost
    solve_time: float


@dataclass
class SolverTimeSummary:
    """Summary of total time per solver."""
    solver_name: str
    total_time: float
    num_graphs_solved: int
    avg_time_per_graph: float


@dataclass
class Metrics:
    """Calculated metrics for experiment results."""
    feasibility_rate: float
    success_rate: float
    dominance_score: float
    avg_relative_gap: float
    std_relative_gap: float  # Standard deviation of relative gaps (consistency metric)
    num_feasible: int
    num_success: int


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
CONFIG = ExperimentConfig(
    vertex_counts=[6,8,11,13,15],
    penalty_methods=['exact','lucas'], # 'exact','lucas'
    num_reads=100,
    seed=42,
    use_openjij=True,
    use_qwavesampler=True,
    use_simulated_bifurcation=True,
    qwavesampler_types=['path'],  # 'path'/'sa'
    use_synthetic_dataset=True,  # Set to False to skip synthetic datasets
    use_real_world_dataset=True,  # Set to True to include real-world graphs
    success_gap_threshold=0.05,
    verbose=True
)


# =============================================================================
# CONSTANTS
# =============================================================================
# CSV field names
AGGREGATED_FIELDS = [
    'solver_name', 'sampler_type', 'dataset_name', 'num_vertices', 'num_graphs', 'density', 'penalty_mode',
    'feasibility_rate', 'success_rate', 'dominance_score', 'avg_relative_gap', 'std_relative_gap',
    'num_feasible', 'num_success', 'total_time'
]

DETAILED_FIELDS = [
    'solver_name', 'sampler_type', 'num_vertices', 'graph_id', 'num_edges', 'penalty_mode',
    'mu_thermometer', 'mu_bijective', 'energy', 'minla_cost',
    'best_known_cost', 'is_feasible', 'solve_time'
]

REAL_WORLD_FIELDS = [
    'solver_name', 'sampler_type', 'graph_name', 'num_vertices', 'num_edges', 'penalty_mode',
    'mu_thermometer', 'mu_bijective', 'is_feasible', 'objective_value',
    'spectral_cost', 'successive_augmentation_cost', 'local_search_cost',
    'best_known_cost', 'relative_gap', 'solve_time'
]

SOLVER_TIME_FIELDS = ['solver_name', 'total_time', 'num_graphs_solved', 'avg_time_per_graph']


# =============================================================================
# HELPER CLASSES
# =============================================================================
class SolverTimeTracker:
    """Track and manage solver execution times."""
    
    def __init__(self, solver_names: list[str]):
        self.times = {
            name: {'total_time': 0.0, 'num_graphs': 0}
            for name in solver_names
        }
    
    def add_time(self, solver_name: str, time: float) -> None:
        """Add execution time for a solver."""
        self.times[solver_name]['total_time'] += time
        self.times[solver_name]['num_graphs'] += 1
    
    def get_summaries(self) -> list[SolverTimeSummary]:
        """Get time summaries for all solvers."""
        summaries = []
        for solver_name, data in self.times.items():
            num_graphs = data['num_graphs']
            avg_time = data['total_time'] / num_graphs if num_graphs > 0 else 0.0
            summaries.append(SolverTimeSummary(
                solver_name=solver_name,
                total_time=data['total_time'],
                num_graphs_solved=num_graphs,
                avg_time_per_graph=avg_time
            ))
        return summaries
    
    def log_summary(self) -> None:
        """Log time summary for all solvers."""
        print("\n" + "=" * 60)
        print("SOLVER TIME SUMMARY")
        print("=" * 60)
        for solver_name in sorted(self.times.keys()):
            data = self.times[solver_name]
            num_graphs = data['num_graphs']
            avg_time = data['total_time'] / num_graphs if num_graphs > 0 else 0.0
            print(f"{solver_name}:")
            print(f"  Total time: {data['total_time']:.2f}s")
            print(f"  Graphs solved: {num_graphs}")
            print(f"  Avg time per graph: {avg_time:.4f}s")
        print("=" * 60)


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
    
    # Calculate standard deviation of relative gaps
    std_gap = statistics.stdev(relative_gaps) if len(relative_gaps) > 1 else 0.0
    
    return Metrics(
        feasibility_rate=num_feasible / num_graphs,
        success_rate=num_success / num_graphs,
        dominance_score=num_dominant / num_feasible if num_feasible > 0 else 0.0,
        avg_relative_gap=sum(relative_gaps) / len(relative_gaps) if relative_gaps else float('inf'),
        std_relative_gap=std_gap,
        num_feasible=num_feasible,
        num_success=num_success
    )


def init_solvers(config: ExperimentConfig) -> dict[str, object]:
    """Initialize solvers based on configuration."""
    solvers = {}
    if config.use_openjij:
        solvers['OpenJij'] = OpenJijSolver()
    if config.use_qwavesampler:
        for sampler_type in config.qwavesampler_types:
            solver = QWaveSamplerSolver()
            solver.configure(sampler_type=sampler_type)
            solvers[f'QWaveSampler_{sampler_type}'] = solver
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
    solver_name: str,
    sampler_type: str
) -> tuple[GraphResult, DetailedResult, float]:
    """
    Process a single graph with the given solver.
    
    Returns:
        Tuple of (GraphResult, DetailedResult, best_cost)
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
        sampler_type=sampler_type,
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


def process_real_world_graph(
    graph_data: dict,
    solver: object,
    penalty_mode: str,
    solver_name: str,
    sampler_type: str
) -> RealWorldResult:
    """
    Process a single real-world graph and return individual results.
    
    Returns:
        RealWorldResult with all baseline comparisons
    """
    graph_name = graph_data.get('name', f"graph_{graph_data['id']}")
    num_vertices = graph_data['num_vertices']
    edges = graph_data['edges']
    
    # Get baseline results from preprocessed data
    spectral_cost = graph_data['spectral_cost']
    sa_cost = graph_data['successive_augmentation_cost']
    local_search_cost = graph_data['local_search_cost']
    best_known_cost = graph_data['best_cost']
    
    graph = build_graph(edges, num_vertices)
    num_edges = graph.number_of_edges()
    
    mu_thermo, mu_bijec = get_penalty_bounds(graph, penalty_mode)
    
    start_time = time.time()
    result = solver.solve(graph)
    solve_time = time.time() - start_time
    
    # Calculate objective value (MINLA cost)
    objective_value = None
    relative_gap = None
    
    if result.is_feasible:
        objective_value = calculate_min_linear_arrangement(graph, result.ordering)
        if best_known_cost > 0:
            relative_gap = (objective_value - best_known_cost) / best_known_cost
    
    return RealWorldResult(
        solver_name=solver_name,
        sampler_type=sampler_type,
        graph_name=graph_name,
        num_vertices=num_vertices,
        num_edges=num_edges,
        penalty_mode=penalty_mode,
        mu_thermometer=mu_thermo,
        mu_bijective=mu_bijec,
        is_feasible=result.is_feasible,
        objective_value=objective_value,
        spectral_cost=spectral_cost,
        successive_augmentation_cost=sa_cost,
        local_search_cost=local_search_cost,
        best_known_cost=best_known_cost,
        relative_gap=relative_gap,
        solve_time=solve_time
    )


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
    print(f"Use synthetic dataset: {config.use_synthetic_dataset}")
    print(f"Use real-world dataset: {config.use_real_world_dataset}")
    print(f"Solvers: {list(solvers.keys())}")
    print("=" * 60)


def log_metrics(metrics: Metrics, num_graphs: int, total_time: float) -> None:
    """Log metrics for a solver run."""
    tqdm.write(f"      Feasibility: {metrics.feasibility_rate*100:.1f}% ({metrics.num_feasible}/{num_graphs})")
    tqdm.write(f"      Success rate: {metrics.success_rate*100:.1f}% ({metrics.num_success}/{num_graphs})")
    tqdm.write(f"      Dominance: {metrics.dominance_score*100:.1f}%")
    tqdm.write(f"      Avg relative gap: {metrics.avg_relative_gap*100:.2f}%")
    tqdm.write(f"      Std relative gap: {metrics.std_relative_gap*100:.2f}%")
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


def collect_datasets(config: ExperimentConfig, base_dir: str) -> list[tuple[str, any, str]]:
    """
    Collect all datasets to process based on configuration.
    
    Returns:
        List of tuples: (dataset_type, vertex_label, dataset_path)
    """
    datasets_to_process = []
    dataset_dir = os.path.join(base_dir, config.dataset_dir)
    
    # Add synthetic datasets
    if config.use_synthetic_dataset:
        for n in config.vertex_counts:
            dataset_path = os.path.join(dataset_dir, f'quantum_n{n}.pkl')
            if os.path.exists(dataset_path):
                datasets_to_process.append(('synthetic', n, dataset_path))
            else:
                print(f"Warning: Synthetic dataset not found: {dataset_path}")
    
    # Add real-world dataset if enabled
    if config.use_real_world_dataset:
        real_world_path = os.path.join(base_dir, config.real_world_dataset_path)
        if os.path.exists(real_world_path):
            datasets_to_process.append(('real_world', 'mixed', real_world_path))
        else:
            print(f"Warning: Real-world dataset not found: {real_world_path}")
    
    return datasets_to_process


def process_synthetic_dataset(
    graphs_data: list[dict],
    n: int,
    solver: object,
    solver_name: str,
    penalty_mode: str,
    config: ExperimentConfig,
    time_tracker: SolverTimeTracker,
    sampler_type: str
) -> tuple[list[GraphResult], list[DetailedResult], list[int], float]:
    """
    Process a synthetic dataset with the given solver.
    
    Returns:
        Tuple of (graph_results, detailed_results, best_costs, total_time)
    """
    graph_results: list[GraphResult] = []
    detailed_results: list[DetailedResult] = []
    best_costs: list[int] = []
    total_time = 0.0
    
    for graph_data in tqdm(graphs_data, desc="    Graphs", leave=False, position=1):
        graph_n = graph_data.get('num_vertices', n)
        
        result, detailed, best_cost = process_single_graph(
            graph_data, graph_n, solver, penalty_mode, solver_name, sampler_type
        )
        graph_results.append(result)
        best_costs.append(best_cost)
        detailed_results.append(detailed)
        total_time += detailed.solve_time
        time_tracker.add_time(solver_name, detailed.solve_time)
    
    return graph_results, detailed_results, best_costs, total_time


def process_realworld_dataset(
    graphs_data: list[dict],
    solver: object,
    solver_name: str,
    penalty_mode: str,
    time_tracker: SolverTimeTracker,
    sampler_type: str
) -> tuple[list[RealWorldResult], float]:
    """
    Process a real-world dataset with the given solver.
    
    Returns:
        Tuple of (real_world_results, total_time)
    """
    real_world_results: list[RealWorldResult] = []
    total_time = 0.0
    
    for graph_data in tqdm(graphs_data, desc="    Graphs", leave=False, position=1):
        result = process_real_world_graph(graph_data, solver, penalty_mode, solver_name, sampler_type)
        real_world_results.append(result)
        total_time += result.solve_time
        time_tracker.add_time(solver_name, result.solve_time)
    
    return real_world_results, total_time


def save_experiment_results(
    experiment_folder: str,
    aggregated_results: list[AggregatedResult],
    detailed_results: list[DetailedResult],
    real_world_results: list[RealWorldResult],
    time_tracker: SolverTimeTracker,
    config: ExperimentConfig
) -> None:
    """Save all experiment results to CSV files."""
    if config.save_aggregated and aggregated_results:
        agg_path = os.path.join(experiment_folder, 'aggregated_results.csv')
        save_results_to_csv(aggregated_results, agg_path, AGGREGATED_FIELDS)
        if config.verbose:
            print(f"\nAggregated results saved to: {agg_path}")
    
    if config.save_detailed and detailed_results:
        detail_path = os.path.join(experiment_folder, 'detailed_results.csv')
        save_results_to_csv(detailed_results, detail_path, DETAILED_FIELDS)
        if config.verbose:
            print(f"Detailed results saved to: {detail_path}")
    
    if real_world_results:
        real_world_path = os.path.join(experiment_folder, 'real_world_results.csv')
        save_results_to_csv(real_world_results, real_world_path, REAL_WORLD_FIELDS)
        if config.verbose:
            print(f"Real-world results saved to: {real_world_path}")
    
    # Save solver time summary
    time_summaries = time_tracker.get_summaries()
    if time_summaries:
        time_summary_path = os.path.join(experiment_folder, 'solver_times.csv')
        save_results_to_csv(time_summaries, time_summary_path, SOLVER_TIME_FIELDS)
        if config.verbose:
            print(f"Solver time summary saved to: {time_summary_path}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment(config: ExperimentConfig = None) -> tuple[list[AggregatedResult], list[DetailedResult], list[RealWorldResult]]:
    """Run quantum experiment on all datasets."""
    if config is None:
        config = CONFIG
    
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, config.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    solvers = init_solvers(config)
    if not solvers:
        print("No solvers enabled. Please enable at least one solver in config.")
        return [], [], []
    
    log_config(config, solvers)
    
    # Initialize result containers
    aggregated_results: list[AggregatedResult] = []
    detailed_results: list[DetailedResult] = []
    real_world_results: list[RealWorldResult] = []
    time_tracker = SolverTimeTracker(list(solvers.keys()))
    
    # Collect all datasets to process
    datasets_to_process = collect_datasets(config, base_dir)
    
    if not datasets_to_process:
        print("No datasets enabled. Please enable at least one dataset type in config.")
        return [], [], []
    
    total_configs = len(datasets_to_process) * len(config.penalty_methods) * len(solvers)
    pbar_configs = tqdm(total=total_configs, desc="Configurations", position=0)
    
    for dataset_type, vertex_label, dataset_path in datasets_to_process:
        dataset = load_dataset(dataset_path)
        graphs_data = dataset['graphs']
        num_graphs = len(graphs_data)
        
        # Get dataset info
        if dataset_type == 'synthetic':
            n = vertex_label
            density = dataset['density']
            dataset_name = f"synthetic_n{n}"
        else:  # real_world
            n = dataset.get('avg_vertices', 0)
            density = dataset.get('avg_density', 0.0)
            dataset_name = "real_world"
        
        if config.verbose:
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Testing dataset: {dataset_name}, {num_graphs} graphs, density={density:.3f}")
            tqdm.write(f"{'='*60}")
        
        for penalty_mode in config.penalty_methods:
            if config.verbose:
                tqdm.write(f"\n  Penalty mode: {penalty_mode}")
            
            for solver_name, solver in solvers.items():
                pbar_configs.set_description(f"{dataset_name}, {penalty_mode}, {solver_name}")
                
                if config.verbose:
                    tqdm.write(f"    Solver: {solver_name}...")
                
                solver.configure(penalty_mode=penalty_mode)
                
                # Get sampler_type for QWaveSampler, "N/A" for others
                sampler_type = getattr(solver, 'sampler_type', 'N/A')
                
                if dataset_type == 'real_world':
                    # Process real-world dataset (no aggregation)
                    rw_results, total_time = process_realworld_dataset(
                        graphs_data, solver, solver_name, penalty_mode, time_tracker, sampler_type
                    )
                    real_world_results.extend(rw_results)
                    
                    if config.verbose:
                        tqdm.write(f"      Total time: {total_time:.2f}s")
                
                else:
                    # Process synthetic dataset (with aggregation)
                    graph_results, details, best_costs, total_time = process_synthetic_dataset(
                        graphs_data, n, solver, solver_name, penalty_mode, config, time_tracker, sampler_type
                    )
                    detailed_results.extend(details)
                    
                    metrics = calculate_metrics(graph_results, best_costs, config.success_gap_threshold)
                    
                    aggregated_results.append(AggregatedResult(
                        solver_name=solver_name,
                        sampler_type=sampler_type,
                        dataset_name=dataset_name,
                        num_vertices=n if dataset_type == 'synthetic' else int(n),
                        num_graphs=num_graphs,
                        density=density,
                        penalty_mode=penalty_mode,
                        feasibility_rate=metrics.feasibility_rate,
                        success_rate=metrics.success_rate,
                        dominance_score=metrics.dominance_score,
                        avg_relative_gap=metrics.avg_relative_gap,
                        std_relative_gap=metrics.std_relative_gap,
                        num_feasible=metrics.num_feasible,
                        num_success=metrics.num_success,
                        total_time=total_time
                    ))
                    
                    if config.verbose:
                        log_metrics(metrics, num_graphs, total_time)
                
                pbar_configs.update(1)
    
    pbar_configs.close()
    
    # Log solver time summary
    if config.verbose:
        time_tracker.log_summary()
    
    # Save results to timestamped folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(results_dir, f'quantum_experiment_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)
    
    save_experiment_results(
        experiment_folder, aggregated_results, detailed_results,
        real_world_results, time_tracker, config
    )
    
    if config.verbose:
        print(f"\nAll results saved in folder: {experiment_folder}")
    
    return aggregated_results, detailed_results, real_world_results


if __name__ == "__main__":
    run_experiment()
