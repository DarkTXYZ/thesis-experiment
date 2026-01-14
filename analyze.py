import pandas as pd
import json
import os

def load_experiment_results(csv_path: str) -> pd.DataFrame:
    """Load quantum experiment results from CSV."""
    return pd.read_csv(csv_path)

def load_baseline_results(json_path: str) -> pd.DataFrame:
    """Load baseline results from JSON."""
    with open(json_path) as f:
        baseline = json.load(f)
    
    df = pd.DataFrame(baseline)[['name', 'spectral_cost', 'successive_augmentation_cost', 'local_search_cost', 'best_cost']]
    df = df.rename(columns={'name': 'graph_name'})
    return df

def compare_results(experiment_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison dataframe between experiment and baseline results."""
    # Pivot experiment results: one row per graph+penalty_mode, columns for each solver
    pivot_df = experiment_df.pivot_table(
        index=['graph_name', 'penalty_mode'], 
        columns='solver_name', 
        values='minla_cost',
        aggfunc='first'
    ).reset_index()
    
    # Merge with baseline
    comparison = pivot_df.merge(baseline_df, on='graph_name')
    
    # Reorder columns
    solver_cols = [col for col in pivot_df.columns if col not in ['graph_name', 'penalty_mode']]
    baseline_cols = ['spectral_cost', 'successive_augmentation_cost', 'local_search_cost', 'best_cost']
    comparison = comparison[['graph_name', 'penalty_mode'] + solver_cols + baseline_cols]
    
    return comparison

def main():
    base_dir = os.path.dirname(__file__)
    
    # Find latest experiment results
    results_dir = os.path.join(base_dir, 'Results')
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('quantum_experiment') and f.endswith('.csv')]
    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest_csv)
    
    # Load data
    experiment_df = load_experiment_results(csv_path)
    baseline_df = load_baseline_results(os.path.join(base_dir, 'Dataset', 'quantum_dataset', 'minla_summary.json'))
    
    # Generate comparison
    comparison = compare_results(experiment_df, baseline_df)
    
    print(f"=== Experiment Results: {latest_csv} ===\n")
    
    # Show full comparison
    print("=== Full Comparison ===\n")
    print(comparison.to_string(index=False))
    
    # Show lucas penalty results only (typically better)
    print("\n\n=== Lucas Penalty Results vs Baselines ===\n")
    lucas_df = comparison[comparison['penalty_mode'] == 'lucas'].copy()
    print(lucas_df.to_string(index=False))
    
    # Calculate gap from best known
    print("\n\n=== Gap from Best Known (Lucas Penalty) ===\n")
    solver_cols = [col for col in lucas_df.columns if col not in ['graph_name', 'penalty_mode', 'spectral_cost', 'successive_augmentation_cost', 'local_search_cost', 'best_cost']]
    
    gap_df = lucas_df[['graph_name']].copy()
    for solver in solver_cols:
        gap_df[f'{solver}_gap'] = lucas_df[solver] - lucas_df['best_cost']
    gap_df['best_cost'] = lucas_df['best_cost']
    print(gap_df.to_string(index=False))

if __name__ == "__main__":
    main()
