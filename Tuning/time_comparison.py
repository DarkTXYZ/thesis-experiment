"""
Compare PathIntegral vs openjij SQA on 30-vertex dataset graphs.
Runs a quick benchmark on a small graph subset and prints timing/energy stats.
"""

import os
import sys
import pickle
import time
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import openjij as oj
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler

# Add parent directory to path to import Utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla


# Paths
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset")

# Experiment configuration
SEED = 42
NUM_VERTICES = 30
MAX_GRAPHS = 1
NUM_READS = 10
NUM_SWEEPS = 1000
BETA_MIN = 1e-6
BETA_MAX = 100


def read_dataset_for_vertices(num_vertices: int):
    """Load dataset pickle file for a specific vertex count."""
    filepath = os.path.join(DATASET_PATH, f"quantum_n{num_vertices}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, "rb") as f:
        return pickle.load(f)


def convert_graph_data_to_nx(graph_data):
    """Convert dataset graph dictionary to a NetworkX graph."""
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(graph_data["num_vertices"]))
    graph_nx.add_edges_from(graph_data["edges"])
    return graph_nx


def decode_solution(raw_sample: Dict, n: int) -> Tuple[np.ndarray, bool]:
    """Decode binary variables X[u][k] into ordering and feasibility."""
    sol = np.array(
        [[raw_sample.get(f"X[{u}][{k}]", 0) for k in range(n)] for u in range(n)],
        dtype=int,
    )
    is_feasible = check_feasibility(sol, n)
    ordering = np.sum(sol, axis=1)
    return ordering, is_feasible


def check_feasibility(sol: np.ndarray, n: int) -> bool:
    """Check if decoded solution satisfies monotonicity and label coverage."""
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return False
    labels = set(np.sum(sol, axis=1))
    return labels == set(range(1, n + 1))


def build_custom_schedules(num_sweeps: int, beta_min: float, beta_max: float):
    """Build compatible custom schedules for both samplers."""
    hp_field = np.linspace(beta_min, beta_max, num=num_sweeps)
    hd_field = np.linspace(beta_max, beta_min, num=num_sweeps)

    s_values = np.linspace(0.0, 1.0, num=num_sweeps)
    beta_values = np.linspace(beta_min, beta_max, num=num_sweeps)
    sqa_schedule = [[float(s), float(beta), 1] for s, beta in zip(s_values, beta_values)]

    return hp_field, hd_field, sqa_schedule


def run_path_integral_solver(
    bqm,
    hp_field,
    hd_field,
    num_reads: int = NUM_READS,
    num_sweeps: int = NUM_SWEEPS,
) -> Tuple[float, Dict]:
    """Run PathIntegral with explicit custom Hp/Hd schedule."""
    solver = PathIntegralAnnealingSampler()
    sampleset = solver.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_schedule_type="custom",
        Hp_field=hp_field,
        Hd_field=hd_field,
    )
    best = sampleset.first
    return best.energy, best.sample


def run_sqa_solver(
    bqm,
    schedule,
    num_reads: int = NUM_READS,
    num_sweeps: int = NUM_SWEEPS,
    seed: int = SEED,
) -> Tuple[float, Dict]:
    """Run openjij SQA with explicit custom schedule."""
    solver = oj.SQASampler()
    sampleset = solver.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        schedule=schedule,
        sparse=True,
        seed=seed,
    )
    best = sampleset.first
    return best.energy, best.sample


def run_experiment():
    """Compare PathIntegral and SQA on a small subset of 30-vertex graphs."""
    np.random.seed(SEED)

    print(f"Loading {NUM_VERTICES}-vertex graph dataset...")
    dataset = read_dataset_for_vertices(NUM_VERTICES)
    graphs = dataset["graphs"]
    num_vertices = dataset["num_vertices"]

    selected_graphs = graphs[:MAX_GRAPHS]
    processed_count = len(selected_graphs)
    if processed_count == 0:
        raise ValueError("No graphs available for experiment.")

    print(f"Loaded {len(graphs)} graphs with {num_vertices} vertices")
    print(f"Testing first {processed_count} graph(s)")
    print(f"Settings: num_reads={NUM_READS}, num_sweeps={NUM_SWEEPS}, beta=({BETA_MIN}, {BETA_MAX})")
    print("Schedules: PathIntegral(custom Hp/Hd), SQA(custom schedule)\n")

    hp_field, hd_field, sqa_schedule = build_custom_schedules(NUM_SWEEPS, BETA_MIN, BETA_MAX)

    all_rows = []
    solver_stats = {
        "PathIntegral": {"feasible": 0, "times": [], "approx_ratios": []},
        "SQA": {"feasible": 0, "times": [], "approx_ratios": []},
    }

    for graph_id, graph in enumerate(selected_graphs):
        graph_nx = convert_graph_data_to_nx(graph)
        n = graph_nx.number_of_nodes()
        m = graph_nx.number_of_edges()

        bqm = minla.generate_bqm_instance(graph_nx)
        optimal_cost = graph.get("optimal_cost", None)

        t0 = time.time()
        pi_energy, pi_sample = run_path_integral_solver(
            bqm=bqm,
            hp_field=hp_field,
            hd_field=hd_field,
        )
        pi_time = time.time() - t0

        pi_ordering, pi_feasible = decode_solution(pi_sample, n)
        pi_minla_cost = minla.calculate_min_linear_arrangement(graph_nx, pi_ordering) if pi_feasible else None
        pi_approx_ratio = (pi_minla_cost / optimal_cost) if (pi_feasible and optimal_cost) else None

        if pi_feasible:
            solver_stats["PathIntegral"]["feasible"] += 1
            solver_stats["PathIntegral"]["approx_ratios"].append(pi_approx_ratio)
        solver_stats["PathIntegral"]["times"].append(pi_time)

        all_rows.append(
            {
                "graph_id": graph_id,
                "n": n,
                "m": m,
                "solver": "PathIntegral",
                "energy": pi_energy,
                "feasible": pi_feasible,
                "minla_cost": pi_minla_cost,
                "optimal_cost": optimal_cost,
                "approx_ratio": pi_approx_ratio,
                "time_s": round(pi_time, 3),
            }
        )

        t0 = time.time()
        sqa_energy, sqa_sample = run_sqa_solver(
            bqm=bqm,
            schedule=sqa_schedule,
        )
        sqa_time = time.time() - t0

        sqa_ordering, sqa_feasible = decode_solution(sqa_sample, n)
        sqa_minla_cost = minla.calculate_min_linear_arrangement(graph_nx, sqa_ordering) if sqa_feasible else None
        sqa_approx_ratio = (sqa_minla_cost / optimal_cost) if (sqa_feasible and optimal_cost) else None

        if sqa_feasible:
            solver_stats["SQA"]["feasible"] += 1
            solver_stats["SQA"]["approx_ratios"].append(sqa_approx_ratio)
        solver_stats["SQA"]["times"].append(sqa_time)

        all_rows.append(
            {
                "graph_id": graph_id,
                "n": n,
                "m": m,
                "solver": "SQA",
                "energy": sqa_energy,
                "feasible": sqa_feasible,
                "minla_cost": sqa_minla_cost,
                "optimal_cost": optimal_cost,
                "approx_ratio": sqa_approx_ratio,
                "time_s": round(sqa_time, 3),
            }
        )

        speedup = pi_time / sqa_time if sqa_time > 0 else 0.0
        print(
            f"Graph {graph_id:2d}: PI E={pi_energy:8.2f} ({pi_time:5.2f}s) | "
            f"SQA E={sqa_energy:8.2f} ({sqa_time:5.2f}s) | "
            f"Speedup={speedup:.2f}x | PI Feas={pi_feasible} | SQA Feas={sqa_feasible}"
        )

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT SUMMARY: PathIntegral vs SQA on {num_vertices}-vertex graphs")
    print(f"{'=' * 80}")

    for solver_name, stats in solver_stats.items():
        feasibility_rate = stats["feasible"] / processed_count
        avg_time = float(np.mean(stats["times"]))
        std_time = float(np.std(stats["times"]))
        min_time = float(np.min(stats["times"]))
        max_time = float(np.max(stats["times"]))
        avg_approx = float(np.mean(stats["approx_ratios"])) if stats["approx_ratios"] else None

        print(f"\n{solver_name}:")
        print(f"  Feasibility Rate:  {feasibility_rate:.1%} ({stats['feasible']}/{processed_count})")
        print(f"  Avg Time:          {avg_time:.3f}s (+/-{std_time:.3f}s)")
        print(f"  Min/Max Time:      {min_time:.3f}s / {max_time:.3f}s")
        if avg_approx is not None:
            print(f"  Avg Approx Ratio:  {avg_approx:.4f}")

    pi_times = solver_stats["PathIntegral"]["times"]
    sqa_times = solver_stats["SQA"]["times"]
    speedups = [pi_t / sqa_t if sqa_t > 0 else 0.0 for pi_t, sqa_t in zip(pi_times, sqa_times)]

    print("\nSpeedup (PathIntegral / SQA):")
    print(f"  Mean:    {float(np.mean(speedups)):.2f}x")
    print(f"  Median:  {float(np.median(speedups)):.2f}x")
    print(f"  Min:     {float(np.min(speedups)):.2f}x")
    print(f"  Max:     {float(np.max(speedups)):.2f}x")

    faster_solver = "SQA" if np.mean(sqa_times) < np.mean(pi_times) else "PathIntegral"
    print(f"\n  Overall: {faster_solver} is faster on average")
    print(f"{'=' * 80}\n")

    # Keep results in memory only (no CSV output).
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    _ = run_experiment()
