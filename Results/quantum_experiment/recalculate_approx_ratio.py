"""
Recalculate approximation ratios with different lower bounds without re-running solver.

This script allows you to:
1. Load existing experiment results (avg_minla_cost and lower_bound)
2. Update lower bounds using different calculation methods
3. Recalculate approximation ratios based on new lower bounds
4. Save the updated results to a new CSV file
"""

import os
import sys
import pickle
import pandas as pd
import networkx as nx
from typing import Dict, List

# Add parent directories to path to import Utils and Baseline
# Script location: Results/quantum_experiment/recalculate_approx_ratio.py
# Need to go up to: thesis-experiment/ (the project root)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Results/quantum_experiment
parent1 = os.path.dirname(script_dir)  # Results
parent2 = os.path.dirname(parent1)  # thesis-experiment (project root)
project_root = parent2

sys.path.insert(0, project_root)

# Verify path is correct
if not os.path.exists(os.path.join(project_root, 'Utils')):
    raise RuntimeError(f"Could not find Utils directory. Project root: {project_root}")

import Utils.MinLA as minla
from Baseline.lower_bound import calculate_lower_obj_bound

DATASET_PATH = os.path.join(project_root, "Dataset", "quantum_dataset")
RESULTS_DIR = os.path.join(project_root, "Results", "quantum_experiment")


def read_dataset():
    """Read all pickle files in DATASET_PATH"""
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data

    return datasets


def convert_graph_data_to_nx(graph_data):
    """Convert graph data dict to NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(range(graph_data['num_vertices']))
    G.add_edges_from(graph_data['edges'])
    return G


def load_results(csv_path: str) -> pd.DataFrame:
    """Load experiment results from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded results from {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def get_graphs_by_instance(datasets: Dict, n: int, graph_id: int) -> Dict:
    """Retrieve a specific graph instance from dataset"""
    graphs = datasets[n]['graphs']
    if graph_id < len(graphs):
        return graphs[graph_id]
    return None


def recalculate_with_new_lower_bounds(
    results_df: pd.DataFrame,
    datasets: Dict,
    lower_bound_method: str = 'default'
) -> pd.DataFrame:
    """
    Recalculate approximation ratios with new lower bounds.
    
    Args:
        results_df: DataFrame with existing results
        datasets: Loaded dataset dictionary
        lower_bound_method: Method to use for calculating lower bound
                           'default': use calculate_lower_obj_bound
    
    Returns:
        Updated DataFrame with new lower bounds and approx ratios
    """
    df = results_df.copy()
    new_lower_bounds = []
    new_approx_ratios = []
    cnt = 0
    
    for idx, row in df.iterrows():
        n = row['n']
        graph_id = row['graph_id']
        avg_minla_cost = row['avg_minla_cost']
        feasible = row['feasible']
        sampler = row.get('sampler', '').lower()
        
        # Skip rows with randomsampler - keep original values
        if 'randomsampler' in sampler:
            new_lower_bounds.append(row['lower_bound'])
            new_approx_ratios.append(row['approx_ratio'])
            continue
        
        # Retrieve the graph
        graph = get_graphs_by_instance(datasets, n, graph_id)
        
        if graph is None:
            print(f"Warning: Could not find graph with n={n}, graph_id={graph_id}")
            new_lower_bounds.append(row['lower_bound'])
            new_approx_ratios.append(row['approx_ratio'])
            continue
        
        # Convert to NetworkX graph
        G = convert_graph_data_to_nx(graph)
        
        # Calculate new lower bound
        new_lb = calculate_lower_obj_bound(G)
        new_lower_bounds.append(new_lb)
        
        # Recalculate approx ratio
        if feasible and pd.notna(avg_minla_cost):
            new_ratio = avg_minla_cost / new_lb
            new_approx_ratios.append(new_ratio)
        else:
            new_approx_ratios.append(None)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows...")

    # Update DataFrame
    df['lower_bound'] = new_lower_bounds
    df['approx_ratio'] = new_approx_ratios
    
    return df


def save_updated_results(
    df: pd.DataFrame,
    output_path: str = None,
    suffix: str = "_updated"
) -> str:
    """
    Save updated results to CSV.
    
    Args:
        df: Updated DataFrame
        output_path: Path to save to (if None, generates from timestamp)
        suffix: Suffix to append to filename
    
    Returns:
        Path to saved file
    """
    import time
    
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            RESULTS_DIR,
            f"quantum_experiment{suffix}_{timestamp}.csv"
        )
    
    os.makedirs(os.path.dirname(output_path) or RESULTS_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nUpdated results saved to {output_path}")
    
    return output_path


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics from results"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for n in sorted(df['n'].unique()):
        subset = df[df['n'] == n]
        feasible_subset = subset[subset['feasible'] == True]
        
        feasibility_rate = len(feasible_subset) / len(subset)
        
        if len(feasible_subset) > 0:
            avg_approx_ratio = feasible_subset['approx_ratio'].mean()
            min_approx_ratio = feasible_subset['approx_ratio'].min()
            max_approx_ratio = feasible_subset['approx_ratio'].max()
        else:
            avg_approx_ratio = None
            min_approx_ratio = None
            max_approx_ratio = None
        
        print(f"\nVertices: {n}")
        print(f"  Instances: {len(subset)}")
        print(f"  Feasible: {len(feasible_subset)} ({feasibility_rate:.2%})")
        print(f"  Avg Approx Ratio: {avg_approx_ratio:.4f}" if avg_approx_ratio else "  Avg Approx Ratio: N/A")
        print(f"  Min Approx Ratio: {min_approx_ratio:.4f}" if min_approx_ratio else "  Min Approx Ratio: N/A")
        print(f"  Max Approx Ratio: {max_approx_ratio:.4f}" if max_approx_ratio else "  Max Approx Ratio: N/A")


def main():
    import sys
    
    # Configuration
    results_csv = "Results/quantum_experiment/quantum_experiment_linear_20260401_085012.csv"
    
    # Check if custom CSV path provided as argument
    if len(sys.argv) > 1:
        results_csv = sys.argv[1]
    
    print("="*70)
    print("RECALCULATE APPROXIMATION RATIOS")
    print("="*70)
    
    # Load results
    results_df = load_results(results_csv)
    
    # Load dataset
    print("\nLoading dataset...")
    datasets = read_dataset()
    print(f"Loaded {len(datasets)} vertex groups from dataset")
    
    # Recalculate with new lower bounds
    print("\nRecalculating approximation ratios with new lower bounds...")
    updated_df = recalculate_with_new_lower_bounds(results_df, datasets)
    
    # Print summary statistics
    print_summary_stats(updated_df)
    
    # Save updated results
    output_csv = save_updated_results(updated_df)
    
    # Show comparison of first few rows
    print("\n" + "="*70)
    print("COMPARISON: OLD vs NEW LOWER BOUNDS")
    print("="*70)
    comparison = pd.DataFrame({
        'old_lb': results_df['lower_bound'],
        'new_lb': updated_df['lower_bound'],
        'old_ratio': results_df['approx_ratio'],
        'new_ratio': updated_df['approx_ratio']
    })
    print(comparison.head(10))
    
    print(f"\n{'='*70}")
    print(f"Updated results available at: {output_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
