"""Find the best (beta_min, beta_max) pair, searched directly over [0, 1].

The per-instance beta ranges elsewhere in this repo come from default_beta_range()
on the un-normalized bqm rescaled by 1/normalize_scale, which produces a ~500x-wide
range (e.g. 0.66 to 343 for one graph in this dataset). That width is what forced
hd_scale down to near-zero just to keep the mirrored Hd_field from overwhelming the
anneal (see git history of this file). Since the bqm is normalized, its biases are
O(1), so beta values of that same order are the physically relevant range to search -
this sweeps (beta_min, beta_max) directly over [0, 1] per graph instead of deriving
them from the raw-bias heuristic. With betas already this small, the natural mirrored
Hd_field (= reversed Hp_field) stays well-scaled too, so hd_scale_multiplier is fixed
at 1 (no extra scaling) rather than searched.
"""
import os
import sys
import time
from itertools import combinations

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuning_normalized_bqm_based_on_instance as base

CSV_PATH = os.path.join(base.PARENT_DIR, "beta_range_comparison.csv")

# Fixed, not searched: pin every other knob so the comparison isolates beta_min/beta_max.
base.NUM_SWEEPS = 1000
base.NUM_SWEEPS_PER_BETA = 2
SCHEDULE_TYPE = "linear"
HD_SCALE_MULTIPLIER = 1  # identity - no extra scaling of Hd_field, see module docstring.

BETA_BREAKPOINTS = [1.0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BETA_RANGES = list(combinations(BETA_BREAKPOINTS, 2))  # all (beta_min, beta_max) with beta_min < beta_max


def evaluate(solver, graph_cache, beta_min, beta_max):
    graphs_feasible = 0
    approx_ratios = []
    t0 = time.time()

    for graph in graph_cache:
        G, n, lower_bound, bqm = graph["G"], graph["n"], graph["lower_bound"], graph["bqm"]
        sample_kwargs = base.build_sample_kwargs(
            "PA", beta_min, beta_max, SCHEDULE_TYPE, HD_SCALE_MULTIPLIER, None, None,
        )

        best_cost = None
        for seed in base.SEEDS:
            sampleset = solver.sample(bqm, seed=seed, **sample_kwargs)
            for sample in sampleset.samples():
                ordering, is_feasible = base.decode_solution(sample, n)
                if is_feasible:
                    cost = base.calculate_min_linear_arrangement(G, ordering)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
        if best_cost is not None:
            graphs_feasible += 1
            approx_ratios.append(best_cost / lower_bound)

    elapsed = time.time() - t0
    mean_approx_ratio = (
        sum(approx_ratios) / len(approx_ratios) if approx_ratios else base.INFEASIBLE_APPROX_RATIO
    )
    return graphs_feasible, mean_approx_ratio, elapsed


def run_comparison():
    graph_cache = base.load_graph_cache()
    solver = base.SOLVERS["PA"]

    print(f"{len(graph_cache)} graphs, num_sweeps={base.NUM_SWEEPS}, "
          f"schedule_type={SCHEDULE_TYPE}, hd_scale={HD_SCALE_MULTIPLIER} (fixed), "
          f"{len(BETA_RANGES)} beta ranges")

    rows = []
    for beta_min, beta_max in BETA_RANGES:
        graphs_feasible, mean_approx_ratio, elapsed = evaluate(solver, graph_cache, beta_min, beta_max)
        print(f"beta=({beta_min:.1f},{beta_max:.1f}) graphs_feasible={graphs_feasible}/{len(graph_cache)} "
              f"mean_approx_ratio={mean_approx_ratio:.4f} time={elapsed:.2f}s")
        rows.append(dict(
            beta_min=beta_min,
            beta_max=beta_max,
            graphs_feasible=graphs_feasible,
            graphs_total=len(graph_cache),
            mean_approx_ratio=mean_approx_ratio,
            time_s=round(elapsed, 3),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nResults saved to {CSV_PATH}")

    best = df.sort_values(["graphs_feasible", "mean_approx_ratio"], ascending=[False, True]).iloc[0]
    print(f"\nBest beta range: beta_min={best['beta_min']}, beta_max={best['beta_max']} "
          f"(graphs_feasible={best['graphs_feasible']}/{len(graph_cache)}, "
          f"mean_approx_ratio={best['mean_approx_ratio']:.4f})")

    return df


if __name__ == "__main__":
    run_comparison()
