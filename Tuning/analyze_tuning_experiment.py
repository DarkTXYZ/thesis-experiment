import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PARENT_DIR, "Results/tuning_experiment")
DETAILED_CSV = os.path.join(RESULTS_DIR, "tuning_experiment_detailed.csv")

ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")
BEST_CONFIGS_CSV = os.path.join(ANALYSIS_DIR, "best_configs.csv")
BEST_CONFIGS_OVERALL_CSV = os.path.join(ANALYSIS_DIR, "best_configs_overall.csv")

GROUP_COLS = ["solver", "n", "num_sweeps", "beta_min", "beta_max", "beta_schedule_type", "qubits_per_chain"]
GROUP_COLS_OVERALL = ["solver", "num_sweeps", "beta_min", "beta_max", "beta_schedule_type", "qubits_per_chain"]
TOP_N = 5


def load_detailed():
    if not os.path.exists(DETAILED_CSV):
        raise FileNotFoundError(f"No detailed results found at {DETAILED_CSV}")
    df = pd.read_csv(DETAILED_CSV)
    # Rows written before qubits_per_chain was tracked ran with the sampler
    # default (1) - backfill so old and new rows group together correctly.
    if "qubits_per_chain" not in df.columns:
        df["qubits_per_chain"] = 1
    else:
        df["qubits_per_chain"] = df["qubits_per_chain"].fillna(1).astype(int)
    return df


def aggregate(df):
    """Aggregate across graphs/seeds per (solver, n, config). Unlike the
    summary CSV written by tuning_experiment.py, this keeps n separate so
    results aren't blended across graph sizes."""
    rows = []
    for keys, group in df.groupby(GROUP_COLS):
        feasible_runs = group[group["feasible"] == True]
        rows.append({
            **dict(zip(GROUP_COLS, keys)),
            "feasibility_rate": len(feasible_runs) / len(group),
            "mean_approx_ratio": feasible_runs["approx_ratio"].mean() if len(feasible_runs) > 0 else None,
            "mean_time_s": group["time_s"].mean(),
            "num_runs": len(group),
        })
    return pd.DataFrame(rows)


def aggregate_overall(df):
    """Aggregate across graphs/seeds/n per (solver, config), pooling all
    graph sizes together. Use this to pick one config per solver rather
    than one per (solver, n)."""
    rows = []
    for keys, group in df.groupby(GROUP_COLS_OVERALL):
        feasible_runs = group[group["feasible"] == True]
        rows.append({
            **dict(zip(GROUP_COLS_OVERALL, keys)),
            "feasibility_rate": len(feasible_runs) / len(group),
            "mean_approx_ratio": feasible_runs["approx_ratio"].mean() if len(feasible_runs) > 0 else None,
            "mean_time_s": group["time_s"].mean(),
            "num_runs": len(group),
        })
    return pd.DataFrame(rows)


def print_overview(df):
    print("=" * 78)
    print("OVERVIEW")
    print("=" * 78)
    print(f"Rows: {len(df)}")
    print(f"Solvers: {sorted(df['solver'].unique())}")
    print(f"Graph sizes (n): {sorted(df['n'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"num_sweeps grid: {sorted(df['num_sweeps'].unique())}")
    print(f"beta_schedule_type: {sorted(df['beta_schedule_type'].unique())}")
    print(f"qubits_per_chain grid: {sorted(df['qubits_per_chain'].unique())}")
    overall_feasible = df["feasible"].sum()
    print(f"Overall feasibility rate: {overall_feasible / len(df):.1%}")
    print()


def print_best_configs(agg):
    print("=" * 78)
    print(f"TOP {TOP_N} CONFIGS PER (solver, n)")
    print("=" * 78)
    for solver in sorted(agg["solver"].unique()):
        for n in sorted(agg[agg["solver"] == solver]["n"].unique()):
            subset = agg[(agg["solver"] == solver) & (agg["n"] == n)].copy()
            subset["mean_approx_ratio"] = subset["mean_approx_ratio"].fillna(float("inf"))
            subset = subset.sort_values(
                by=["feasibility_rate", "mean_approx_ratio", "mean_time_s"],
                ascending=[False, True, True],
            ).head(TOP_N)

            print(f"\n{solver} | n={n}")
            print("-" * 78)
            for _, row in subset.iterrows():
                ratio_str = "N/A" if row["mean_approx_ratio"] == float("inf") else f"{row['mean_approx_ratio']:.4f}"
                print(
                    f"  sweeps={int(row['num_sweeps']):<5} "
                    f"beta=({row['beta_min']:.2e},{row['beta_max']:.2e}) "
                    f"type={row['beta_schedule_type']:<9} "
                    f"qpc={int(row['qubits_per_chain']):<3} | "
                    f"feas_rate={row['feasibility_rate']:.0%}  "
                    f"approx_ratio={ratio_str:<8} "
                    f"time={row['mean_time_s']:.3f}s"
                )
    print()


def print_parameter_effects(agg):
    print("=" * 78)
    print("MARGINAL PARAMETER EFFECTS (averaged over all other config values)")
    print("=" * 78)
    for solver in sorted(agg["solver"].unique()):
        solver_agg = agg[agg["solver"] == solver]
        print(f"\n{solver}")
        for param in ["num_sweeps", "beta_schedule_type", "beta_min", "beta_max", "qubits_per_chain"]:
            print(f"  by {param}:")
            grouped = solver_agg.groupby(param).agg(
                feasibility_rate=("feasibility_rate", "mean"),
                mean_approx_ratio=("mean_approx_ratio", "mean"),
                mean_time_s=("mean_time_s", "mean"),
            )
            for value, row in grouped.iterrows():
                ratio_str = f"{row['mean_approx_ratio']:.4f}" if pd.notna(row["mean_approx_ratio"]) else "N/A"
                print(
                    f"    {str(value):<12} feas_rate={row['feasibility_rate']:.1%}  "
                    f"approx_ratio={ratio_str:<8} time={row['mean_time_s']:.3f}s"
                )
    print()


def save_best_configs_csv(agg):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    rows = []
    for solver in sorted(agg["solver"].unique()):
        for n in sorted(agg[agg["solver"] == solver]["n"].unique()):
            subset = agg[(agg["solver"] == solver) & (agg["n"] == n)].copy()
            subset["mean_approx_ratio"] = subset["mean_approx_ratio"].fillna(float("inf"))
            subset = subset.sort_values(
                by=["feasibility_rate", "mean_approx_ratio", "mean_time_s"],
                ascending=[False, True, True],
            ).head(TOP_N)
            rows.append(subset)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(BEST_CONFIGS_CSV, index=False)
    print(f"Best configs saved to {BEST_CONFIGS_CSV}")


def print_best_configs_overall(agg_overall):
    print("=" * 78)
    print(f"TOP {TOP_N} CONFIGS PER SOLVER (pooled across n)")
    print("=" * 78)
    for solver in sorted(agg_overall["solver"].unique()):
        subset = agg_overall[agg_overall["solver"] == solver].copy()
        subset["mean_approx_ratio"] = subset["mean_approx_ratio"].fillna(float("inf"))
        subset = subset.sort_values(
            by=["feasibility_rate", "mean_approx_ratio", "mean_time_s"],
            ascending=[False, True, True],
        ).head(TOP_N)

        print(f"\n{solver}")
        print("-" * 78)
        for _, row in subset.iterrows():
            ratio_str = "N/A" if row["mean_approx_ratio"] == float("inf") else f"{row['mean_approx_ratio']:.4f}"
            print(
                f"  sweeps={int(row['num_sweeps']):<5} "
                f"beta=({row['beta_min']:.2e},{row['beta_max']:.2e}) "
                f"type={row['beta_schedule_type']:<9} "
                f"qpc={int(row['qubits_per_chain']):<3} | "
                f"feas_rate={row['feasibility_rate']:.0%}  "
                f"approx_ratio={ratio_str:<8} "
                f"time={row['mean_time_s']:.3f}s  "
                f"n_runs={int(row['num_runs'])}"
            )
    print()


def save_best_configs_overall_csv(agg_overall):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    rows = []
    for solver in sorted(agg_overall["solver"].unique()):
        subset = agg_overall[agg_overall["solver"] == solver].copy()
        subset["mean_approx_ratio"] = subset["mean_approx_ratio"].fillna(float("inf"))
        subset = subset.sort_values(
            by=["feasibility_rate", "mean_approx_ratio", "mean_time_s"],
            ascending=[False, True, True],
        ).head(TOP_N)
        rows.append(subset)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(BEST_CONFIGS_OVERALL_CSV, index=False)
    print(f"Best overall configs saved to {BEST_CONFIGS_OVERALL_CSV}")


def plot_beta_heatmaps(agg):
    """One figure per (solver, n, qubits_per_chain): rows = beta_schedule_type,
    cols = num_sweeps. Each cell is a beta_min x beta_max heatmap colored by
    mean_approx_ratio, annotated with the feasibility rate. Cells with 0%
    feasibility have no approx_ratio to color by, so they're drawn gray rather
    than left blank - blank would be indistinguishable from "not run".
    qubits_per_chain is faceted into separate files (not another grid axis)
    since SA only ever has one value (1) and PIA has four - a shared grid
    layout would leave SA's figures three-quarters empty."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cmap = sns.color_palette("Blues", as_cmap=True)
    cmap.set_bad("#d9d9d9")

    for solver in sorted(agg["solver"].unique()):
        for n in sorted(agg[agg["solver"] == solver]["n"].unique()):
            qpc_vals = sorted(agg[(agg["solver"] == solver) & (agg["n"] == n)]["qubits_per_chain"].unique())
            for qpc in qpc_vals:
                subset = agg[(agg["solver"] == solver) & (agg["n"] == n) & (agg["qubits_per_chain"] == qpc)]
                sweeps_vals = sorted(subset["num_sweeps"].unique())
                schedule_vals = sorted(subset["beta_schedule_type"].unique())
                beta_min_vals = sorted(subset["beta_min"].unique())
                beta_max_vals = sorted(subset["beta_max"].unique())

                fig, axes = plt.subplots(
                    len(schedule_vals), len(sweeps_vals),
                    figsize=(4.2 * len(sweeps_vals), 3.6 * len(schedule_vals)),
                    squeeze=False,
                )

                vmax = subset["mean_approx_ratio"].max()
                for i, schedule in enumerate(schedule_vals):
                    for j, sweeps in enumerate(sweeps_vals):
                        ax = axes[i][j]
                        cell = subset[(subset["beta_schedule_type"] == schedule) & (subset["num_sweeps"] == sweeps)]
                        pivot_ratio = cell.pivot(index="beta_min", columns="beta_max", values="mean_approx_ratio")
                        pivot_feas = cell.pivot(index="beta_min", columns="beta_max", values="feasibility_rate")
                        pivot_ratio = pivot_ratio.reindex(index=beta_min_vals, columns=beta_max_vals)
                        pivot_feas = pivot_feas.reindex(index=beta_min_vals, columns=beta_max_vals)

                        sns.heatmap(
                            pivot_ratio, ax=ax, cmap=cmap, vmin=1.0, vmax=vmax,
                            cbar=(j == len(sweeps_vals) - 1),
                            linewidths=0.5, linecolor="white",
                            cbar_kws={"label": "mean approx ratio"} if j == len(sweeps_vals) - 1 else None,
                        )

                        for row_idx, beta_min in enumerate(beta_min_vals):
                            for col_idx, beta_max in enumerate(beta_max_vals):
                                feas = pivot_feas.iloc[row_idx, col_idx]
                                if pd.isna(feas):
                                    continue
                                ratio_known = pd.notna(pivot_ratio.iloc[row_idx, col_idx])
                                text_color = "white" if (ratio_known and pivot_ratio.iloc[row_idx, col_idx] > (1 + vmax) / 2) else "black"
                                ax.text(
                                    col_idx + 0.5, row_idx + 0.5, f"{feas:.0%}",
                                    ha="center", va="center", color=text_color, fontsize=9,
                                )

                        ax.set_title(f"sweeps={sweeps}, {schedule}", fontsize=10)
                        ax.set_xlabel("beta_max")
                        ax.set_ylabel("beta_min" if j == 0 else "")

                fig.suptitle(
                    f"{solver} | n={n}, qubits_per_chain={qpc} — "
                    f"color=mean approx ratio, label=feasibility rate (gray=0% feasible)",
                    fontsize=12,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                out_path = os.path.join(PLOTS_DIR, f"heatmap_{solver}_n{n}_qpc{qpc}.png")
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"Saved {out_path}")


def plot_num_sweeps_trend(agg):
    """Best (over beta grid + schedule type) feasibility rate and approx ratio
    vs num_sweeps, faceted by graph size n."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    best_per_sweeps = (
        agg.sort_values("mean_approx_ratio")
        .groupby(["solver", "n", "num_sweeps"], as_index=False)
        .agg(feasibility_rate=("feasibility_rate", "max"), mean_approx_ratio=("mean_approx_ratio", "min"))
    )

    for metric, ylabel in [("feasibility_rate", "Best feasibility rate"), ("mean_approx_ratio", "Best mean approx ratio")]:
        n_vals = sorted(best_per_sweeps["n"].unique())
        fig, axes = plt.subplots(1, len(n_vals), figsize=(4.2 * len(n_vals), 3.6), squeeze=False, sharey=True)
        axes = axes[0]
        for ax, n in zip(axes, n_vals):
            data = best_per_sweeps[best_per_sweeps["n"] == n]
            sns.lineplot(data=data, x="num_sweeps", y=metric, hue="solver", marker="o", ax=ax)
            ax.set_title(f"n={n}")
            ax.set_ylabel(ylabel if ax is axes[0] else "")
        fig.suptitle(f"{ylabel} vs num_sweeps by graph size", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = os.path.join(PLOTS_DIR, f"trend_{metric}_vs_num_sweeps.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_qubits_per_chain_trend(agg):
    """Best (over beta grid, schedule type, num_sweeps) feasibility rate and
    approx ratio vs qubits_per_chain, faceted by graph size n. SA only has the
    default value (1) and shows as a flat reference line; PIA is the one
    where this axis is actually swept."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    best_per_qpc = (
        agg.sort_values("mean_approx_ratio")
        .groupby(["solver", "n", "qubits_per_chain"], as_index=False)
        .agg(feasibility_rate=("feasibility_rate", "max"), mean_approx_ratio=("mean_approx_ratio", "min"))
    )

    for metric, ylabel in [("feasibility_rate", "Best feasibility rate"), ("mean_approx_ratio", "Best mean approx ratio")]:
        n_vals = sorted(best_per_qpc["n"].unique())
        fig, axes = plt.subplots(1, len(n_vals), figsize=(4.2 * len(n_vals), 3.6), squeeze=False, sharey=True)
        axes = axes[0]
        for ax, n in zip(axes, n_vals):
            data = best_per_qpc[best_per_qpc["n"] == n]
            sns.lineplot(data=data, x="qubits_per_chain", y=metric, hue="solver", marker="o", ax=ax)
            ax.set_title(f"n={n}")
            ax.set_ylabel(ylabel if ax is axes[0] else "")
        fig.suptitle(f"{ylabel} vs qubits_per_chain by graph size", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = os.path.join(PLOTS_DIR, f"trend_{metric}_vs_qubits_per_chain.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    df = load_detailed()
    agg = aggregate(df)
    agg_overall = aggregate_overall(df)

    print_overview(df)
    print_best_configs(agg)
    print_best_configs_overall(agg_overall)
    print_parameter_effects(agg)
    save_best_configs_csv(agg)
    save_best_configs_overall_csv(agg_overall)
    plot_beta_heatmaps(agg)
    plot_num_sweeps_trend(agg)
    plot_qubits_per_chain_trend(agg)


if __name__ == "__main__":
    main()
