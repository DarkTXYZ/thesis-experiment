"""
Analyze quantum experiment aggregated results.
Display metrics, comparisons, and visualizations.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional


def load_latest_aggregated_results(results_dir: str = "Results") -> Optional[pd.DataFrame]:
    """Load the most recent aggregated results CSV."""
    results_path = Path(results_dir)
    
    # Look for experiment folders
    experiment_folders = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('quantum_experiment_')]
    
    if not experiment_folders:
        # Fallback: look for old-style CSV files
        csv_files = list(results_path.glob("quantum_experiment_aggregated_*.csv"))
        if not csv_files:
            print(f"No aggregated results found in {results_dir}")
            return None
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    else:
        # Get the most recent folder
        latest_folder = max(experiment_folders, key=lambda x: x.stat().st_mtime)
        latest_file = latest_folder / 'aggregated_results.csv'
        
        if not latest_file.exists():
            print(f"No aggregated_results.csv found in {latest_folder}")
            return None
    
    print(f"Loading: {latest_file}\n")
    
    df = pd.read_csv(latest_file)
    return df


def display_summary_table(df: pd.DataFrame) -> None:
    """Display a formatted summary table."""
    print("=" * 140)
    print("QUANTUM EXPERIMENT AGGREGATED RESULTS")
    print("=" * 140)
    
    # Format the dataframe for better display
    display_df = df.copy()
    display_df['feasibility_rate'] = (display_df['feasibility_rate'] * 100).round(1).astype(str) + '%'
    display_df['success_rate'] = (display_df['success_rate'] * 100).round(1).astype(str) + '%'
    display_df['dominance_score'] = (display_df['dominance_score'] * 100).round(1).astype(str) + '%'
    display_df['avg_relative_gap'] = (display_df['avg_relative_gap'] * 100).round(2).astype(str) + '%'
    
    # Add std_relative_gap if it exists in the dataframe
    if 'std_relative_gap' in display_df.columns:
        display_df['std_relative_gap'] = (display_df['std_relative_gap'] * 100).round(2).astype(str) + '%'
    
    display_df['total_time'] = (display_df['total_time'] / 60).round(2).astype(str) + ' min'
    
    # Reorder columns
    cols = ['solver_name', 'num_vertices', 'penalty_mode', 'num_graphs', 
            'feasibility_rate', 'success_rate', 'dominance_score', 
            'avg_relative_gap']
    
    if 'std_relative_gap' in display_df.columns:
        cols.append('std_relative_gap')
    
    cols.append('total_time')
    display_df = display_df[cols]
    
    print(display_df.to_string(index=False))
    print("=" * 140)


def display_solver_comparison(df: pd.DataFrame) -> None:
    """Compare solvers across all configurations."""
    print("\n" + "=" * 80)
    print("SOLVER COMPARISON BY VERTEX COUNT AND PENALTY MODE")
    print("=" * 80)
    
    for n in sorted(df['num_vertices'].unique()):
        print(f"\n{n} vertices:")
        subset = df[df['num_vertices'] == n].copy()
        
        for penalty in sorted(subset['penalty_mode'].unique()):
            print(f"\n  Penalty mode: {penalty}")
            penalty_subset = subset[subset['penalty_mode'] == penalty].copy()
            
            for _, row in penalty_subset.iterrows():
                std_gap_str = f"Std: {row['std_relative_gap']*100:6.2f}% | " if 'std_relative_gap' in row else ""
                print(f"    {row['solver_name']:20s} | "
                      f"Feasibility: {row['feasibility_rate']*100:5.1f}% | "
                      f"Success: {row['success_rate']*100:5.1f}% | "
                      f"Dominance: {row['dominance_score']*100:5.1f}% | "
                      f"Gap: {row['avg_relative_gap']*100:6.2f}% | "
                      f"{std_gap_str}"
                      f"Time: {row['total_time']/60:7.2f} min")


def display_best_configurations(df: pd.DataFrame) -> None:
    """Show best performing configurations for each metric."""
    print("\n" + "=" * 80)
    print("BEST PERFORMING CONFIGURATIONS")
    print("=" * 80)
    
    metrics = {
        'Highest Feasibility Rate': ('feasibility_rate', False),
        'Highest Success Rate': ('success_rate', False),
        'Highest Dominance Score': ('dominance_score', False),
        'Lowest Avg Gap': ('avg_relative_gap', True),
        'Fastest Total Time': ('total_time', True)
    }
    
    for metric_name, (column, ascending) in metrics.items():
        # Filter out inf values for gap
        if column == 'avg_relative_gap':
            valid_df = df[df[column] != float('inf')]
        else:
            valid_df = df
        
        if valid_df.empty:
            continue
            
        best = valid_df.nsmallest(1, column) if ascending else valid_df.nlargest(1, column)
        
        if not best.empty:
            row = best.iloc[0]
            value = row[column]
            if column in ['feasibility_rate', 'success_rate', 'dominance_score', 'avg_relative_gap']:
                value_str = f"{value*100:.2f}%"
            elif column == 'total_time':
                value_str = f"{value/60:.2f} min"
            else:
                value_str = f"{value}"
            
            print(f"\n{metric_name}: {value_str}")
            print(f"  Solver: {row['solver_name']}, Vertices: {row['num_vertices']}, "
                  f"Penalty: {row['penalty_mode']}")


def display_penalty_comparison(df: pd.DataFrame) -> None:
    """Compare exact vs lucas penalty modes."""
    print("\n" + "=" * 80)
    print("PENALTY MODE COMPARISON (EXACT vs LUCAS)")
    print("=" * 80)
    
    for solver in sorted(df['solver_name'].unique()):
        print(f"\n{solver}:")
        solver_df = df[df['solver_name'] == solver]
        
        for n in sorted(solver_df['num_vertices'].unique()):
            subset = solver_df[solver_df['num_vertices'] == n]
            
            exact = subset[subset['penalty_mode'] == 'exact']
            lucas = subset[subset['penalty_mode'] == 'lucas']
            
            if exact.empty or lucas.empty:
                continue
            
            exact_row = exact.iloc[0]
            lucas_row = lucas.iloc[0]
            
            print(f"  n={n:2d}: Exact [Feas: {exact_row['feasibility_rate']*100:5.1f}%, "
                  f"Succ: {exact_row['success_rate']*100:5.1f}%] | "
                  f"Lucas [Feas: {lucas_row['feasibility_rate']*100:5.1f}%, "
                  f"Succ: {lucas_row['success_rate']*100:5.1f}%]")


def display_vertex_scaling(df: pd.DataFrame) -> None:
    """Show how metrics scale with vertex count."""
    print("\n" + "=" * 80)
    print("SCALING WITH VERTEX COUNT")
    print("=" * 80)
    
    for solver in sorted(df['solver_name'].unique()):
        print(f"\n{solver}:")
        solver_df = df[df['solver_name'] == solver]
        
        for penalty in sorted(solver_df['penalty_mode'].unique()):
            print(f"\n  Penalty: {penalty}")
            subset = solver_df[solver_df['penalty_mode'] == penalty].sort_values('num_vertices')
            
            print(f"    {'n':>3s} | {'Feasibility':>11s} | {'Success':>7s} | {'Time (min)':>10s}")
            print(f"    {'-'*3:3s} | {'-'*11:11s} | {'-'*7:7s} | {'-'*10:10s}")
            
            for _, row in subset.iterrows():
                print(f"    {row['num_vertices']:3d} | "
                      f"{row['feasibility_rate']*100:10.1f}% | "
                      f"{row['success_rate']*100:6.1f}% | "
                      f"{row['total_time']/60:10.2f}")


def format_time_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def display_total_time_summary(df: pd.DataFrame) -> None:
    """Display total time used across all experiments in HH:MM:SS format."""
    print("\n" + "=" * 80)
    print("TOTAL TIME SUMMARY")
    print("=" * 80)
    
    # Overall total
    total_seconds = df['total_time'].sum()
    print(f"\nTotal time across all experiments: {format_time_hms(total_seconds)} (HH:MM:SS)")
    print(f"                                    {total_seconds/60:.2f} minutes")
    print(f"                                    {total_seconds/3600:.2f} hours")
    
    # Per solver
    print("\nTime by solver:")
    for solver in sorted(df['solver_name'].unique()):
        solver_total = df[df['solver_name'] == solver]['total_time'].sum()
        percentage = (solver_total / total_seconds * 100) if total_seconds > 0 else 0
        print(f"  {solver:25s}: {format_time_hms(solver_total)} ({solver_total/60:7.2f} min, {percentage:5.1f}%)")
    
    # Per penalty mode
    print("\nTime by penalty mode:")
    for penalty in sorted(df['penalty_mode'].unique()):
        penalty_total = df[df['penalty_mode'] == penalty]['total_time'].sum()
        percentage = (penalty_total / total_seconds * 100) if total_seconds > 0 else 0
        print(f"  {penalty:10s}: {format_time_hms(penalty_total)} ({penalty_total/60:7.2f} min, {percentage:5.1f}%)")
    
    # Per vertex count
    print("\nTime by vertex count:")
    for n in sorted(df['num_vertices'].unique()):
        n_total = df[df['num_vertices'] == n]['total_time'].sum()
        percentage = (n_total / total_seconds * 100) if total_seconds > 0 else 0
        print(f"  n={n:2d}: {format_time_hms(n_total)} ({n_total/60:7.2f} min, {percentage:5.1f}%)")
    
    print("=" * 80)


def main():
    """Main analysis function."""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "Results")
    
    # Load data
    df = load_latest_aggregated_results(results_dir)
    if df is None:
        return
    
    # Display analyses
    display_summary_table(df)
    display_solver_comparison(df)
    display_best_configurations(df)
    display_penalty_comparison(df)
    display_vertex_scaling(df)
    display_total_time_summary(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
