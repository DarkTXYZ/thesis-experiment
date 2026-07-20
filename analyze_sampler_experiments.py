import glob
import os

import pandas as pd

RESULT_DIRS = {
    "PIA": "Results/pia_experiment",
    "SA": "Results/sa_experiment",
}
OUTPUT_PATH = "Results/sampler_experiment_summary.csv"


def latest_csv(result_dir):
    """Return the most recently modified CSV in result_dir."""
    csv_files = glob.glob(os.path.join(result_dir, "*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getmtime)


def summarize(df):
    """Compute feasibility rate, approx-ratio mean/std, and avg runtime per (solver, n)."""
    rows = []

    for solver in sorted(df["solver"].unique()):
        df_solver = df[df["solver"] == solver]

        for n in sorted(df_solver["n"].unique()):
            df_n = df_solver[df_solver["n"] == n]
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
        df_solver = summary_df[summary_df["solver"] == solver]
        print(f"\n{solver}")
        print("-" * 80)
        print(f"{'n':<5} {'Feasible':<12} {'F_N,s':<10} {'mu_rho':<12} {'sigma_rho':<12} {'Avg Time (s)':<14}")
        print("-" * 80)
        for _, row in df_solver.iterrows():
            mu_str = f"{row['mean_approx_ratio_ub']:.4f}" if pd.notna(row["mean_approx_ratio_ub"]) else "N/A"
            sigma_str = f"{row['std_approx_ratio_ub']:.4f}" if pd.notna(row["std_approx_ratio_ub"]) else "N/A"
            print(
                f"{row['n']:<5} {row['num_feasible']}/{row['num_graphs']:<9} "
                f"{row['feasibility_rate']:<10.3f} {mu_str:<12} {sigma_str:<12} {row['avg_time_s']:<14.4f}"
            )


def main():
    all_summaries = []

    for label, result_dir in RESULT_DIRS.items():
        csv_path = latest_csv(result_dir)
        if csv_path is None:
            print(f"[{label}] No CSV files found in {result_dir}, skipping.")
            continue

        print(f"[{label}] Using latest result file: {csv_path}")
        df = pd.read_csv(csv_path)
        summary_df = summarize(df)
        print_summary(summary_df)
        all_summaries.append(summary_df)

    if not all_summaries:
        print("No results to summarize.")
        return

    combined = pd.concat(all_summaries, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved combined summary -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
