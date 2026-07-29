"""Find the best (beta_min, beta_max) pair, searched directly as fixed breakpoints.

The per-instance beta ranges elsewhere in this repo come from default_beta_range()
on the un-normalized bqm rescaled by 1/normalize_scale (e.g. 0.66 to 343 for the
n=10 graph this script evaluates). Normalizing shrinks biases by the same factor
it shrinks beta_min/beta_max apart from - it does NOT mean beta itself should be
O(1): since acceptance probability goes as exp(-beta*deltaE), shrinking deltaE
(smaller normalized biases) requires a proportionally LARGER beta to reach the same
"coldness", not a smaller one. Measured default_beta_range()-derived estimates across
this dataset's graphs range ~340-1071, growing with graph size. BETA_BREAKPOINTS below
brackets that same order of magnitude as a direct sanity check against the per-instance
heuristic, instead of assuming a scale a priori.

NUM_SWEEPS=200 is likewise measured, not guessed: at num_sweeps=10 (this script's
original value) every beta range tested gave 0/N feasible reads - zero signal to
compare on. Scanning num_sweeps on the n=10 graph at beta_max=343 showed feasibility
climbing 0%(10)->12%(50)->34%(100)->62%(200)->93%(1000), while the best solution found
already plateaus by num_sweeps=200. 200 is the point where there's enough signal to
compare beta ranges without paying for sweeps beyond the quality plateau.
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
base.NUM_SWEEPS = 200
base.NUM_SWEEPS_PER_BETA = 10
SCHEDULE_TYPE = "geometric"
HD_SCALE_MULTIPLIER = 1  # identity - no extra scaling of Hd_field, see module docstring.

BETA_BREAKPOINTS = [512]
# BETA_RANGES = list(combinations(BETA_BREAKPOINTS, 2))  # all (beta_min, beta_max) with beta_min < beta_max
BETA_RANGES = [(1, beta) for beta in BETA_BREAKPOINTS]  # all (0, beta_max) pairs


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
            
        break

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
