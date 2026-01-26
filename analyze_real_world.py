"""
Analyze quantum real-world experiment results.
Display individual graph results and comparisons with baselines.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional


def load_latest_real_world_results(results_dir: str = "Results") -> Optional[pd.DataFrame]:
    """Load the most recent real-world results CSV."""
    results_path = Path(results_dir)
    
    # Look for experiment folders
    experiment_folders = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('quantum_experiment_')]
    
    if not experiment_folders:
        print(f"No experiment folders found in {results_dir}")
        return None
    
    # Get the most recent folder
    latest_folder = max(experiment_folders, key=lambda x: x.stat().st_mtime)
    latest_file = latest_folder / 'real_world_results.csv'
    
    if not latest_file.exists():
        print(f"No real_world_results.csv found in {latest_folder}")
        return None
    
    print(f"Loading: {latest_file}\n")
    
    df = pd.read_csv(latest_file)
    return df


def display_summary_table(df: pd.DataFrame) -> None:
    """Display a formatted summary table."""
    print("=" * 160)
    print("REAL-WORLD EXPERIMENT RESULTS")
    print("=" * 160)
    
    # Format the dataframe for better display
    display_df = df.copy()
    
    # Format relative gap as percentage
    display_df['relative_gap'] = display_df['relative_gap'].apply(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
    )
    display_df['solve_time'] = (display_df['solve_time']).round(3).astype(str) + 's'
    
    # Select and reorder columns
    cols = ['solver_name', 'graph_name', 'num_vertices', 'num_edges', 'penalty_mode',
            'is_feasible', 'objective_value', 'best_known_cost', 'relative_gap', 'solve_time']
    display_df = display_df[cols]
    
    print(display_df.to_string(index=False))
    print("=" * 160)


def display_solver_comparison(df: pd.DataFrame) -> None:
    """Compare solvers for each graph."""
    print("\n" + "=" * 100)
    print("SOLVER COMPARISON BY GRAPH")
    print("=" * 100)
    
    for graph in sorted(df['graph_name'].unique()):
        print(f"\n{graph}:")
        subset = df[df['graph_name'] == graph].copy()
        
        # Get graph info
        n = subset['num_vertices'].iloc[0]
        m = subset['num_edges'].iloc[0]
        best_known = subset['best_known_cost'].iloc[0]
        
        print(f"  Vertices: {n}, Edges: {m}, Best known cost: {best_known}")
        
        for penalty in sorted(subset['penalty_mode'].unique()):
            print(f"\n  Penalty mode: {penalty}")
            penalty_subset = subset[subset['penalty_mode'] == penalty]
            
            for _, row in penalty_subset.iterrows():
                status = "✓ FEASIBLE" if row['is_feasible'] else "✗ INFEASIBLE"
                
                if row['is_feasible']:
                    obj_val = row['objective_value']
                    gap = row['relative_gap'] * 100 if pd.notna(row['relative_gap']) else None
                    
                    if obj_val <= best_known:
                        quality = "★ OPTIMAL/BETTER"
                    elif gap is not None and gap <= 5:
                        quality = "◆ NEAR-OPTIMAL"
                    else:
                        quality = ""
                    
                    gap_str = f"{gap:.2f}%" if gap is not None else "N/A"
                    print(f"    {row['solver_name']:25s} | {status:15s} | Cost: {obj_val:4.0f} | Gap: {gap_str:8s} | {quality}")
                else:
                    print(f"    {row['solver_name']:25s} | {status:15s}")


def display_feasibility_summary(df: pd.DataFrame) -> None:
    """Display feasibility rates by solver."""
    print("\n" + "=" * 80)
    print("FEASIBILITY SUMMARY BY SOLVER")
    print("=" * 80)
    
    for solver in sorted(df['solver_name'].unique()):
        solver_df = df[df['solver_name'] == solver]
        
        for penalty in sorted(solver_df['penalty_mode'].unique()):
            penalty_df = solver_df[solver_df['penalty_mode'] == penalty]
            
            total = len(penalty_df)
            feasible = penalty_df['is_feasible'].sum()
            rate = (feasible / total * 100) if total > 0 else 0
            
            print(f"{solver:25s} | {penalty:10s} | Feasible: {feasible:2d}/{total:2d} ({rate:5.1f}%)")


def display_performance_metrics(df: pd.DataFrame) -> None:
    """Display performance metrics for feasible solutions."""
    print("\n" + "=" * 100)
    print("PERFORMANCE METRICS (FEASIBLE SOLUTIONS ONLY)")
    print("=" * 100)
    
    feasible_df = df[df['is_feasible'] == True].copy()
    
    if feasible_df.empty:
        print("No feasible solutions found!")
        return
    
    for solver in sorted(feasible_df['solver_name'].unique()):
        solver_df = feasible_df[feasible_df['solver_name'] == solver]
        
        for penalty in sorted(solver_df['penalty_mode'].unique()):
            penalty_df = solver_df[solver_df['penalty_mode'] == penalty]
            
            if penalty_df.empty:
                continue
            
            # Calculate metrics
            num_graphs = len(penalty_df)
            num_optimal = (penalty_df['objective_value'] <= penalty_df['best_known_cost']).sum()
            avg_gap = penalty_df['relative_gap'].mean() * 100 if pd.notna(penalty_df['relative_gap']).any() else None
            avg_time = penalty_df['solve_time'].mean()
            
            optimal_rate = (num_optimal / num_graphs * 100) if num_graphs > 0 else 0
            
            print(f"\n{solver} - {penalty}:")
            print(f"  Graphs solved: {num_graphs}")
            print(f"  Optimal/Better: {num_optimal}/{num_graphs} ({optimal_rate:.1f}%)")
            if avg_gap is not None:
                print(f"  Avg relative gap: {avg_gap:.2f}%")
            print(f"  Avg solve time: {avg_time:.3f}s")


def display_baseline_comparison(df: pd.DataFrame) -> None:
    """Compare solver results with baseline methods."""
    print("\n" + "=" * 120)
    print("COMPARISON WITH BASELINE METHODS")
    print("=" * 120)
    
    feasible_df = df[df['is_feasible'] == True].copy()
    
    if feasible_df.empty:
        print("No feasible solutions to compare!")
        return
    
    for graph in sorted(feasible_df['graph_name'].unique()):
        graph_df = feasible_df[feasible_df['graph_name'] == graph]
        
        if graph_df.empty:
            continue
        
        # Get baseline costs
        spectral = graph_df['spectral_cost'].iloc[0]
        sa = graph_df['successive_augmentation_cost'].iloc[0]
        local_search = graph_df['local_search_cost'].iloc[0]
        best_known = graph_df['best_known_cost'].iloc[0]
        
        print(f"\n{graph}:")
        print(f"  Baselines: Spectral={spectral}, SA={sa}, LocalSearch={local_search}, Best={best_known}")
        
        # Find best solver result for each penalty mode
        for penalty in sorted(graph_df['penalty_mode'].unique()):
            penalty_df = graph_df[graph_df['penalty_mode'] == penalty]
            best_solver_row = penalty_df.loc[penalty_df['objective_value'].idxmin()]
            
            solver_cost = best_solver_row['objective_value']
            solver_name = best_solver_row['solver_name']
            
            if solver_cost < best_known:
                status = "★ NEW BEST!"
            elif solver_cost == best_known:
                status = "✓ Matches best"
            else:
                gap = (solver_cost - best_known) / best_known * 100
                status = f"Gap: {gap:.1f}%"
            
            print(f"  {penalty:10s}: Best solver = {solver_name:20s}, Cost = {solver_cost:4.0f} ({status})")


def display_time_analysis(df: pd.DataFrame) -> None:
    """Display time analysis."""
    print("\n" + "=" * 80)
    print("TIME ANALYSIS")
    print("=" * 80)
    
    for solver in sorted(df['solver_name'].unique()):
        solver_df = df[df['solver_name'] == solver]
        total_time = solver_df['solve_time'].sum()
        avg_time = solver_df['solve_time'].mean()
        num_graphs = len(solver_df['graph_name'].unique())
        
        print(f"\n{solver}:")
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Avg time per graph: {avg_time:.3f}s")
        print(f"  Graphs processed: {num_graphs}")


def main():
    """Main analysis function."""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "Results")
    
    # Load data
    df = load_latest_real_world_results(results_dir)
    if df is None:
        return
    
    # Display analyses
    display_summary_table(df)
    display_solver_comparison(df)
    display_feasibility_summary(df)
    display_performance_metrics(df)
    display_baseline_comparison(df)
    display_time_analysis(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
