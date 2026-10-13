import pickle
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "Dataset" / "quantum_dataset"
OUTPUT_DIR = PROJECT_DIR / "Results" / "approx_ratio"

OPTIMAL_FILE = DATASET_DIR / "quantum_n10_d0.5_optimal.pkl"
SAMPLER_FILES = {
    "PIASampler": PROJECT_DIR
    / "Results"
    / "pia_experiment"
    / "pia_experiment_20260819_202756.csv",
    "SASampler": PROJECT_DIR
    / "Results"
    / "sa_experiment"
    / "sa_experiment_20260819_154012.csv",
}


def load_optimal_values() -> pd.DataFrame:
    with OPTIMAL_FILE.open("rb") as file:
        optimal_data = pickle.load(file)

    return pd.DataFrame(
        [
            {
                "graph_id": graph["id"],
                "optimal_value": graph["optimal_value"],
            }
            for graph in optimal_data["graphs"]
        ]
    )


def calculate_ratios() -> Path:
    optimal_values = load_optimal_values()
    result_frames = []

    for sampler_name, sampler_file in SAMPLER_FILES.items():
        sampler_results = pd.read_csv(sampler_file)
        sampler_results = sampler_results[
            (sampler_results["n"] == 10)
            & (sampler_results["penalty"].isin(["lucas", "exact"]))
        ].copy()

        if len(sampler_results) != 200:
            raise ValueError(
                f"{sampler_file} has {len(sampler_results)} matching rows; "
                "expected 200 rows for 100 graphs and 2 penalty modes."
            )

        merged = sampler_results.merge(
            optimal_values,
            on="graph_id",
            how="left",
            validate="many_to_one",
        )
        if merged["optimal_value"].isna().any():
            raise ValueError(f"Missing exact optimum for a graph in {sampler_file}.")

        merged["sampler"] = sampler_name
        merged["density"] = 0.5
        merged["exact_optimal_value"] = merged["optimal_value"]
        merged["exact_optimal_ratio"] = (
            merged["avg_minla_cost"] / merged["exact_optimal_value"]
        )
        result_frames.append(
            merged[
                [
                    "density",
                    "n",
                    "m",
                    "graph_id",
                    "sampler",
                    "penalty",
                    "feasible",
                    "avg_minla_cost",
                    "exact_optimal_value",
                    "exact_optimal_ratio",
                ]
            ]
        )

    output = pd.concat(result_frames, ignore_index=True)
    output = output.sort_values(
        ["sampler", "penalty", "graph_id"], ignore_index=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "approx_ratio_n10_d0.5_20260819.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} rows to {output_path}")
    print(
        output.groupby(["sampler", "penalty"])["exact_optimal_ratio"]
        .agg(["count", "mean", "min", "max"])
        .to_string()
    )
    return output_path


if __name__ == "__main__":
    calculate_ratios()
