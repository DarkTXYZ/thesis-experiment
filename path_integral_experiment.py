from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExperimentConfig:
    seed: int = 42
    verbose: bool = True
    
    vertex_counts: list[int] = field(default_factory=lambda: [])
    penalty_methods: list[str] = field(default_factory=lambda: ['exact', 'lucas'])
    num_reads: int = 10
    success_gap_threshold: float = 0.05 
    
    synthetic_dataset_path: str = "Dataset/quantum_dataset"
    real_world_dataset_path: str = "Dataset/quantum_real_world_dataset/quantum_real_world.pkl"
    
    use_synthetic_dataset: bool = True
    use_real_world_dataset: bool = False
    
    beta_schedule_types: list[str] = field(default_factory=lambda: [
        'default', 'linear', 'geometric' ,'linear_beta', 'exponential', 'power'
        ])
    beta_range: tuple[float, float] = (0.0, 1.0)
    use_auto_beta_range: bool = False
    
    results_dir: str = "Results"
    save_detailed: bool = True
    save_aggregated: bool = True
    

@dataclass
class GraphResult:
    minla_cost: Optional[int]
    is_feasible: bool
    
@dataclass
class DetailedResult:
    solver_name: str
    beta_schedule_type: str
    beta_range: str
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
    solver_name: str
    beta_schedule_type: str
    use_auto_beta_range: bool
    dataset_name: str
    num_vertices: int
    num_graphs: int
    penalty_mode: str
    num_feasible: int
    num_success: int
    feasibility_rate: float
    success_rate: float
    dominance_score: float
    avg_relative_gap: float
    std_relative_gap: float
    total_time: float
    
@dataclass
class RealWorldResult:
    solver_name: str
    beta_schedule_type: str
    beta_range: str 
    graph_name: str
    num_vertices: int
    num_edges: int
    penalty_mode: str
    mu_thermometer: float
    mu_bijective: float
    is_feasible: bool
    objective_value: Optional[int]
    spectral_cost: int
    successive_augmentation_cost: int
    local_search_cost: int
    best_known_cost: int
    relative_gap: Optional[float]
    solve_time: float
    
@dataclass
class SolverTimeSummary:
    solver_name: str
    total_time: float
    num_graphs_solved: int
    avg_time_per_graph: float