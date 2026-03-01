"""
Experiment: Compare built-in 'linear' vs custom 'linear_beta' schedule
in PathIntegralAnnealingSampler across different num_sweeps values.

Custom linear schedule (from QWSamplerSolver):
  Hp_field = hot_beta + (cold_beta - hot_beta) * s   (hot_beta -> cold_beta)
  Hd_field = cold_beta * (1 - s)                     (cold_beta -> 0)
  where s ∈ [0, 1] and beta range is auto-computed from BQM.
"""

import os
import json
import pickle
import time
import warnings
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
import dimod
from dwave.samplers import PathIntegralAnnealingSampler
from dwave.embedding.chain_strength import uniform_torque_compensation

import Utils.MinLA as minla

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
NUM_READS = 10
# NUM_SWEEPS_LIST = [100]
NUM_SWEEPS_LIST = [100, 500, 1000, 2000, 3000]
# NUM_SWEEPS_LIST = [4000, 5000, 6000, 7000, 8000]
GRAPH_SIZES = [40]                     # which quantum_nX.pkl to load
MAX_GRAPHS_PER_SIZE = 1                # only pick one graph
QUBITS_PER_CHAIN = 64
QUBITS_PER_UPDATE = 64
DATASET_DIR = "Dataset/quantum_dataset"
RESULTS_DIR = "Results"

# ──────────────────────────────────────────────────────────────────────────────
# BETA RANGE AUTO-COMPUTATION  (mirrors QWSamplerSolver logic)
# ──────────────────────────────────────────────────────────────────────────────

def default_ising_beta_range(
    h: Dict, J: Dict,
    max_single_qubit_excitation_rate: float = 0.01,
    scale_T_with_N: bool = True
) -> List[float]:
    """Calculate default beta range for Ising model."""
    sum_abs_bias = defaultdict(float, {k: abs(v) for k, v in h.items()})
    min_abs_bias = {}
    if sum_abs_bias:
        min_abs_bias = {k: v for k, v in sum_abs_bias.items() if v != 0}

    for (k1, k2), v in J.items():
        for k in [k1, k2]:
            sum_abs_bias[k] += abs(v)
            if v != 0:
                if k in min_abs_bias:
                    min_abs_bias[k] = min(abs(v), min_abs_bias[k])
                else:
                    min_abs_bias[k] = abs(v)

    if not min_abs_bias:
        warnings.warn("All biases are zero – using [0.1, 1].")
        return [0.1, 1.0]

    max_eff = max(sum_abs_bias.values(), default=0)
    hot_beta = np.log(2) / (2 * max_eff) if max_eff else 1.0

    vals = np.array(list(min_abs_bias.values()), dtype=float)
    min_eff = np.min(vals)
    number_min_gaps = np.sum(min_eff == vals) if scale_T_with_N else 1
    cold_beta = np.log(number_min_gaps / max_single_qubit_excitation_rate) / (2 * min_eff)

    return [hot_beta, cold_beta]


def auto_beta_range(bqm: dimod.BinaryQuadraticModel) -> Tuple[float, float]:
    """Return (hot_beta, cold_beta) computed from the BQM."""
    ising = bqm.spin
    br = default_ising_beta_range(dict(ising.linear), dict(ising.quadratic))
    return tuple(br)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM LINEAR SCHEDULE  (mirrors QWSamplerSolver.generate_linear_schedule)
# ──────────────────────────────────────────────────────────────────────────────

def generate_custom_linear_schedule(
    steps: int,
    beta_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom linear schedule:
      Hp_field: hot_beta → cold_beta  (linearly increasing)
      Hd_field: cold_beta → 0         (linearly decreasing)
    """
    hot_beta, cold_beta = beta_range
    s = np.linspace(0, 1, steps)
    Hp_field = hot_beta + (cold_beta - hot_beta) * s
    Hd_field = cold_beta * (1 - s)
    return Hp_field, Hd_field

# ──────────────────────────────────────────────────────────────────────────────
# DECODE / FEASIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def decode_solution(raw_sample: Dict, n: int) -> Tuple[np.ndarray, bool]:
    sol = np.array([
        [raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
        for u in range(n)
    ], dtype=int)
    is_feasible = check_feasibility(sol, n)
    ordering = np.sum(sol, axis=1)
    return ordering, is_feasible


def check_feasibility(sol: np.ndarray, n: int) -> bool:
    for u in range(n):
        if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
            return False
    labels = set(np.sum(sol, axis=1))
    return labels == set(range(1, n + 1))

# ──────────────────────────────────────────────────────────────────────────────
# SOLVER WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────

def solve_builtin_linear(
    bqm: dimod.BinaryQuadraticModel,
    num_sweeps: int,
    num_reads: int,
    seed: int,
):
    """Solve using the built-in 'linear' beta_schedule_type."""
    sampler = PathIntegralAnnealingSampler()
    return sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_schedule_type='linear',
        # chain_coupler_strength=-uniform_torque_compensation(bqm),
        qubits_per_chain=QUBITS_PER_CHAIN,
        qubits_per_update=QUBITS_PER_UPDATE,
        project_states=(True, True),
    )


def solve_custom_linear(
    bqm: dimod.BinaryQuadraticModel,
    num_sweeps: int,
    num_reads: int,
    seed: int,
):
    """Solve using custom linear schedule (Hp/Hd fields)."""
    sampler = PathIntegralAnnealingSampler()
    beta_range = auto_beta_range(bqm)
    Hp, Hd = generate_custom_linear_schedule(num_sweeps, beta_range)
    return sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_schedule_type='custom',
        Hp_field=Hp,
        Hd_field=Hd,
        # chain_coupler_strength=-uniform_torque_compensation(bqm),
        qubits_per_chain=QUBITS_PER_CHAIN,
        qubits_per_update=QUBITS_PER_UPDATE,
        project_states=(True, True),
    )

# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment():
    rows = []

    for size in GRAPH_SIZES:
        pkl_path = os.path.join(DATASET_DIR, f"quantum_n{size}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[SKIP] {pkl_path} not found")
            continue

        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)

        graphs_data = dataset['graphs'][:MAX_GRAPHS_PER_SIZE]
        print(f"\n{'='*60}")
        print(f" Graph size n={size}  ({len(graphs_data)} graphs)")
        print(f"{'='*60}")

        for gidx, graph_data in enumerate(graphs_data):
            G = nx.Graph()
            G.add_nodes_from(range(graph_data['num_vertices']))
            G.add_edges_from(graph_data['edges'])
            n = G.number_of_nodes()
            m = G.number_of_edges()

            bqm = minla.generate_bqm_instance(G)
            optimal_cost = graph_data.get('optimal_cost', None)

            for num_sweeps in NUM_SWEEPS_LIST:
                for schedule_name, solver_fn in [
                    ('builtin_linear', solve_builtin_linear),
                    ('custom_linear',  solve_custom_linear),
                ]:
                    t0 = time.time()
                    sampleset = solver_fn(bqm, num_sweeps, NUM_READS, SEED)
                    elapsed = time.time() - t0

                    best = sampleset.first
                    ordering, feasible = decode_solution(best.sample, n)
                    minla_cost = (
                        minla.calculate_min_linear_arrangement(G, ordering)
                        if feasible else None
                    )
                    rel_gap = (
                        (minla_cost - optimal_cost) / optimal_cost
                        if (feasible and optimal_cost) else None
                    )

                    row = {
                        'n': n,
                        'm': m,
                        'graph_id': gidx,
                        'schedule': schedule_name,
                        'num_sweeps': num_sweeps,
                        'energy': best.energy,
                        'feasible': feasible,
                        'minla_cost': minla_cost,
                        'optimal_cost': optimal_cost,
                        'relative_gap': rel_gap,
                        'time_s': round(elapsed, 3),
                    }
                    rows.append(row)

                    status = "✓" if feasible else "✗"
                    gap_str = f"{rel_gap:.4f}" if rel_gap is not None else "N/A"
                    print(
                        f"  g{gidx} | {schedule_name:16s} | sweeps={num_sweeps:6d} "
                        f"| {status} E={best.energy:12.2f} | cost={minla_cost} "
                        f"| gap={gap_str} | {elapsed:.2f}s"
                    )

    # ── Save results ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"beta_schedule_experiment_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    summary = (
        df.groupby(['n', 'schedule', 'num_sweeps'])
        .agg(
            feasibility_rate=('feasible', 'mean'),
            avg_energy=('energy', 'mean'),
            avg_cost=('minla_cost', 'mean'),
            avg_gap=('relative_gap', 'mean'),
            avg_time=('time_s', 'mean'),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    return df


if __name__ == "__main__":
    df = run_experiment()
