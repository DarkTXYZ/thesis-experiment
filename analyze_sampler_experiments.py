import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Proactively added RandomSampler for when you run it
RESULT_DIRS = {
    # "PIA": "Results/pia_experiment",
    "SA": "Results/sa_experiment",
    # "Random": "Results/random_experiment", 
}
OUTPUT_PATH = "Results/sampler_experiment_summary.csv"
BOXPLOT_PATH = "Results/plots/results_boxplot.png"


def latest_csv(result_dir):
    """Return the most recently modified CSV in result_dir."""
    csv_files = glob.glob(os.path.join(result_dir, "*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getmtime)


def summarize(df):
    """Compute feasibility rate, approx-ratio mean/std, and avg runtime per (solver, penalty, n)."""
    rows = []

    # Check if 'penalty' exists in the df (in case RandomSampler doesn't have it yet)
    if "penalty" not in df.columns:
        df["penalty"] = "N/A"

    for solver in sorted(df["solver"].unique()):
        df_solver = df[df["solver"] == solver]
        
        for penalty in sorted(df_solver["penalty"].unique()):
            df_penalty = df_solver[df_solver["penalty"] == penalty]

            for n in sorted(df_penalty["n"].unique()):
                df_n = df_penalty[df_penalty["n"] == n]
                
                num_graphs = len(df_n)
                num_feasible = int(df_n["feasible"].sum())
                feasibility_rate = num_feasible / num_graphs

                feasible_df = df_n[df_n["feasible"] == True]
                if len(feasible_df) > 0:
                    mean_ratio = feasible_df["approx_ratio"].mean()
                    std_ratio = feasible_df["approx_ratio"].std() if len(feasible_df) >= 2 else None
                else:
                    mean_ratio = None
                    std_ratio = None

                avg_time = df_n["time_s"].mean()

                rows.append({
                    "solver": solver,
                    "penalty": penalty,
                    "n": n,
                    "num_graphs": num_graphs,
                    "num_feasible": num_feasible,
                    "feasibility_rate": feasibility_rate,
                    "mean_approx_ratio_ub": mean_ratio,
                    "std_approx_ratio_ub": std_ratio,
                    "avg_time_s": avg_time,
                })

    return pd.DataFrame(rows)


def print_summary(summary_df):
    for solver in summary_df["solver"].unique():
        for penalty in summary_df["penalty"].unique():
            df_sp = summary_df[(summary_df["solver"] == solver) & (summary_df["penalty"] == penalty)]
            
            if df_sp.empty:
                continue
                
            print(f"\n{solver} [{penalty.upper()}]")
            print("-" * 80)
            print(f"{'n':<5} {'Feasible':<12} {'F_N,s':<10} {'mu_rho':<12} {'sigma_rho':<12} {'Avg Time (s)':<14}")
            print("-" * 80)
            
            for _, row in df_sp.iterrows():
                mu_str = f"{row['mean_approx_ratio_ub']:.4f}" if pd.notna(row["mean_approx_ratio_ub"]) else "N/A"
                sigma_str = f"{row['std_approx_ratio_ub']:.4f}" if pd.notna(row["std_approx_ratio_ub"]) else "N/A"
                print(
                    f"{row['n']:<5} {row['num_feasible']}/{row['num_graphs']:<9} "
                    f"{row['feasibility_rate']:<10.3f} {mu_str:<12} {sigma_str:<12} {row['avg_time_s']:<14.4f}"
                )


def plot_raw_approx_ratio_boxplot(result_frames):
    """Save raw feasible approximation-ratio distributions for PIA and SA."""
    frames = []
    solver_map = {
        "PathIntegralAnnealingSampler": "PIASampler",
        "SimulatedAnnealingSampler": "SASampler",
    }

    for df in result_frames:
        if "penalty" not in df.columns:
            continue
        frame = df.copy()
        frame["Sampler_Name"] = frame["solver"].map(solver_map)
        frame = frame.dropna(subset=["Sampler_Name", "approx_ratio"])
        frame["Configuration"] = (
            frame["Sampler_Name"] + " (" + frame["penalty"].astype(str).str.capitalize() + ")"
        )
        frames.append(frame)

    if not frames:
        print("\nNo PIA or SA approximation ratios available for boxplot.")
        return

    df_plot = pd.concat(frames, ignore_index=True)
    print(f"Graph sizes: {sorted(df_plot['n'].unique())}")

    hue_order = [
        "SASampler (Exact)",
        "SASampler (Lucas)",
        "PIASampler (Exact)",
        "PIASampler (Lucas)",
    ]
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(figsize=(8, 7))
    sns.boxplot(
        data=df_plot,
        x="n",
        y="approx_ratio",
        hue="Configuration",
        hue_order=hue_order,
        palette="Set2",
        showfliers=True,
        ax=axes,
    )
    # plt.title("Distribution of Approximation Ratio across Graph Sizes", fontsize=14, pad=15)
    plt.ylabel(r"Approximation Ratio ($\rho^{\mathregular{UB}}$)", fontsize=12)
    plt.xlabel(r"Graph Size ($N$)", fontsize=12)
    _, y_max = axes.get_ylim()
    axes.set_ylim(0.95, y_max * 1.05)
    axes.legend(
        title="Sampler and Selection Method",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(BOXPLOT_PATH), exist_ok=True)
    figure.savefig(BOXPLOT_PATH, dpi=300)
    plt.close(figure)
    print(f"Saved raw approximation-ratio boxplot -> {BOXPLOT_PATH}")


def main():
    all_summaries = []
    result_frames = []

    for label, result_dir in RESULT_DIRS.items():
        csv_path = latest_csv(result_dir)
        if csv_path is None:
            print(f"[{label}] No CSV files found in {result_dir}, skipping.")
            continue

        print(f"[{label}] Using latest result file: {csv_path}")
        df = pd.read_csv(csv_path)
        if "penalty" not in df.columns:
            df["penalty"] = "N/A"
        summary_df = summarize(df)
        print_summary(summary_df)
        all_summaries.append(summary_df)
        if label in {"PIA", "SA"}:
            result_frames.append(df)

    if not all_summaries:
        print("\nNo results to summarize.")
        return

    combined = pd.concat(all_summaries, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved combined summary -> {OUTPUT_PATH}")
    plot_raw_approx_ratio_boxplot(result_frames)


if __name__ == "__main__":
    main()