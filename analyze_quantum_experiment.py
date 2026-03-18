import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def analyze_results(file_path):
    """Analyze quantum experiment results from CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return None
    
    # Read CSV
    df = pd.read_csv(file_path)
    print(f"\nLoaded {len(df)} results from {file_path}")
    print(f"{'='*70}\n")
    
    # Analyze by graph size (n)
    print("ANALYSIS BY GRAPH SIZE (n)")
    print("="*70)
    print(f"{'n':<5} {'Graphs':<8} {'Feasible':<12} {'Feas. Rate':<12} {'Avg Ratio':<12} {'Avg Gap':<12}")
    print("-"*70)
    
    summary_data = []
    
    for n in sorted(df['n'].unique()):
        df_n = df[df['n'] == n]
        num_graphs = len(df_n)
        num_feasible = df_n['feasible'].sum()
        feasibility_rate = num_feasible / num_graphs * 100
        
        # Calculate average approximation ratio (only for feasible solutions)
        feasible_df = df_n[df_n['feasible'] == True]
        if len(feasible_df) > 0:
            avg_ratio = feasible_df['approx_ratio'].mean()
        else:
            avg_ratio = None
        
        # Calculate average relative gap (only for feasible solutions)
        if len(feasible_df) > 0:
            avg_gap = feasible_df['relative_gap'].mean()
        else:
            avg_gap = None
        
        # Format output
        ratio_str = f"{avg_ratio:.4f}" if avg_ratio is not None else "N/A"
        gap_str = f"{avg_gap:.4f}" if avg_gap is not None else "N/A"
        
        print(f"{n:<5} {num_graphs:<8} {num_feasible}/{num_graphs:<6} {feasibility_rate:>10.1f}% {ratio_str:<12} {gap_str:<12}")
        
        summary_data.append({
            'n': n,
            'num_graphs': num_graphs,
            'num_feasible': num_feasible,
            'feasibility_rate': feasibility_rate,
            'avg_approx_ratio': avg_ratio,
            'avg_relative_gap': avg_gap,
        })
    
    print("\n" + "="*70)
    print("\nOVERALL STATISTICS")
    print("="*70)
    
    total_graphs = len(df)
    total_feasible = df['feasible'].sum()
    overall_feas_rate = total_feasible / total_graphs * 100
    
    feasible_df = df[df['feasible'] == True]
    if len(feasible_df) > 0:
        overall_avg_ratio = feasible_df['approx_ratio'].mean()
        overall_avg_gap = feasible_df['relative_gap'].mean()
    else:
        overall_avg_ratio = None
        overall_avg_gap = None
    
    print(f"Total graphs tested: {total_graphs}")
    print(f"Total feasible solutions: {total_feasible} ({overall_feas_rate:.1f}%)")
    print(f"Overall avg approx ratio: {overall_avg_ratio:.4f}" if overall_avg_ratio else "Overall avg approx ratio: N/A")
    print(f"Overall avg relative gap: {overall_avg_gap:.4f}" if overall_avg_gap else "Overall avg relative gap: N/A")
    
    print(f"\nAvg solver time: {df['time_s'].mean():.2f}s")
    print(f"Min solver time: {df['time_s'].min():.2f}s")
    print(f"Max solver time: {df['time_s'].max():.2f}s")
    
    print("\n" + "="*70)
    
    return {
        'file_path': file_path,
        'total_graphs': total_graphs,
        'total_feasible': total_feasible,
        'feasibility_rate': overall_feas_rate,
        'avg_approx_ratio': overall_avg_ratio,
        'avg_relative_gap': overall_avg_gap,
        'avg_time': df['time_s'].mean(),
    }


def find_all_quantum_experiments():
    """Find all quantum experiment results files."""
    results_dir = "Results"
    if not os.path.exists(results_dir):
        print(f"Error: '{results_dir}' directory not found")
        return []
    
    # Find all quantum experiment CSV files
    files = [f for f in os.listdir(results_dir) if f.startswith("quantum_experiment_") and f.endswith(".csv")]
    if not files:
        print(f"No quantum_experiment CSV files found in '{results_dir}'")
        return []
    
    # Return full paths sorted by modification time (newest first)
    file_paths = [os.path.join(results_dir, f) for f in files]
    file_paths.sort(key=os.path.getmtime, reverse=True)
    
    return file_paths


def compare_all_experiments():
    """Compare all quantum experiment results and find the best one."""
    file_paths = find_all_quantum_experiments()
    
    if not file_paths:
        print("No quantum experiment files found")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARING ALL QUANTUM EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Found {len(file_paths)} experiment(s)\n")
    
    results_summary = []
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}] {os.path.basename(file_path)}")
        print("-" * 80)
        
        result = analyze_results(file_path)
        if result:
            results_summary.append(result)
    
    # Comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE (All Experiments)")
    print(f"{'='*80}\n")
    
    print(f"{'File':<35} {'Feas%':<8} {'Avg Ratio':<12} {'Avg Gap':<12} {'Avg Time':<10}")
    print("-" * 80)
    
    for result in results_summary:
        filename = os.path.basename(result['file_path'])
        feas_pct = f"{result['feasibility_rate']:.1f}%"
        ratio = f"{result['avg_approx_ratio']:.4f}" if result['avg_approx_ratio'] else "N/A"
        gap = f"{result['avg_relative_gap']:.4f}" if result['avg_relative_gap'] else "N/A"
        time_s = f"{result['avg_time']:.2f}s"
        
        print(f"{filename:<35} {feas_pct:<8} {ratio:<12} {gap:<12} {time_s:<10}")
    
    # Find the best experiment
    print(f"\n{'='*80}")
    print("BEST EXPERIMENT")
    print(f"{'='*80}\n")
    
    # Rank by: 1) Highest feasibility rate, 2) Lowest avg gap, 3) Lowest avg ratio
    best_by_feas = max(results_summary, key=lambda x: x['feasibility_rate'])
    
    feasible_results = [r for r in results_summary if r['total_feasible'] > 0]
    if feasible_results:
        best_by_gap = min(feasible_results, key=lambda x: x['avg_relative_gap'])
        best_by_ratio = min(feasible_results, key=lambda x: x['avg_approx_ratio'])
    else:
        best_by_gap = None
        best_by_ratio = None
    
    print(f"Best feasibility rate: {os.path.basename(best_by_feas['file_path'])}")
    print(f"  Feasibility: {best_by_feas['feasibility_rate']:.1f}%")
    print(f"  Total feasible: {best_by_feas['total_feasible']}/{best_by_feas['total_graphs']}")
    
    if best_by_gap:
        print(f"\nBest avg relative gap: {os.path.basename(best_by_gap['file_path'])}")
        print(f"  Avg gap: {best_by_gap['avg_relative_gap']:.4f}")
        print(f"  Feasibility: {best_by_gap['feasibility_rate']:.1f}%")
    
    if best_by_ratio:
        print(f"\nBest avg approx ratio: {os.path.basename(best_by_ratio['file_path'])}")
        print(f"  Avg ratio: {best_by_ratio['avg_approx_ratio']:.4f}")
        print(f"  Feasibility: {best_by_ratio['feasibility_rate']:.1f}%")
    
    print(f"\n{'='*80}")


def find_latest_results():
    """Find the latest quantum experiment results file."""
    results_dir = "Results"
    if not os.path.exists(results_dir):
        print(f"Error: '{results_dir}' directory not found")
        return None
    
    # Find all quantum experiment CSV files
    files = [f for f in os.listdir(results_dir) if f.startswith("quantum_experiment_") and f.endswith(".csv")]
    if not files:
        print(f"No quantum_experiment CSV files found in '{results_dir}'")
        return None
    
    # Get the latest file by modification time
    latest_file = max(
        [os.path.join(results_dir, f) for f in files],
        key=os.path.getmtime
    )
    return latest_file


if __name__ == "__main__":
    # Set flags here to control behavior
    COMPARE_ALL = True  # Set to True to compare all experiments
    SPECIFIC_FILE = None  # Set to file path to analyze a specific file, e.g., 'Results/quantum_experiment_20260316_145546.csv'
    
    if COMPARE_ALL:
        # Compare all experiments
        compare_all_experiments()
    elif SPECIFIC_FILE and os.path.exists(SPECIFIC_FILE):
        # Analyze specified file
        analyze_results(SPECIFIC_FILE)
    else:
        # Try to use the latest file
        file_paths = find_all_quantum_experiments()
        if file_paths:
            file_path = file_paths[0]  # Use latest
            print(f"Using latest results file: {file_path}\n")
            analyze_results(file_path)
        else:
            print("No quantum experiment files found.")
