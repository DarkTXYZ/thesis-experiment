import pandas as pd
import os

def analyze_results(quantum_file, heuristics_file):
    """Analyze both quantum and heuristics experiments together, grouped by graph size (n)."""
    
    print("\n" + "="*120)
    print("COMBINED EXPERIMENT ANALYSIS - Grouped by Graph Size (n)")
    print("="*120)
    
    # Load both datasets
    df_quantum = None
    df_heuristics = None
    
    if quantum_file and os.path.exists(quantum_file):
        df_quantum = pd.read_csv(quantum_file)
        print(f"\nLoaded {len(df_quantum)} quantum results from {os.path.basename(quantum_file)}")
    
    if heuristics_file and os.path.exists(heuristics_file):
        df_heuristics = pd.read_csv(heuristics_file)
        print(f"Loaded {len(df_heuristics)} heuristics results from {os.path.basename(heuristics_file)}")
    
    if df_quantum is None and df_heuristics is None:
        print("No data files found")
        return
    
    # Combine datasets
    if df_quantum is not None and df_heuristics is not None:
        df = pd.concat([df_quantum, df_heuristics], ignore_index=True)
    elif df_quantum is not None:
        df = df_quantum
    else:
        df = df_heuristics
    
    print("\n" + "="*120)
    print(f"ANALYSIS BY GRAPH SIZE (n) - All Solvers/Heuristics")
    print("="*120)
    
    # Get all unique graph sizes and solvers
    graph_sizes = sorted(df['n'].unique())
    all_solvers = sorted(df['solver'].unique())
    
    for n in graph_sizes:
        df_n = df[df['n'] == n]
        print(f"\n{'─'*120}")
        print(f"Graph Size: n={n}")
        print(f"{'─'*120}")
        print(f"{'Solver/Heuristic':<40} {'Graphs':<8} {'Feasible':<12} {'Feas. %':<10} {'Avg Ratio':<12} {'Avg Time':<10}")
        print("-"*120)
        
        for solver in all_solvers:
            df_solver = df_n[df_n['solver'] == solver]
            
            if len(df_solver) == 0:
                continue
            
            num_graphs = len(df_solver)
            num_feasible = df_solver['feasible'].sum()
            feasibility_rate = num_feasible / num_graphs * 100
            
            feasible_df = df_solver[df_solver['feasible'] == True]
            if len(feasible_df) > 0:
                avg_ratio = feasible_df['approx_ratio'].mean()
            else:
                avg_ratio = None
            
            avg_time = df_solver['time_s'].mean()
            
            ratio_str = f"{avg_ratio:.4f}" if avg_ratio is not None else "N/A"
            time_str = f"{avg_time:.4f}s"
            
            print(f"{solver:<40} {num_graphs:<8} {num_feasible}/{num_graphs:<5} {feasibility_rate:>9.1f}% {ratio_str:<12} {time_str:<10}")
    
    print("\n" + "="*120)
    print(f"OVERALL STATISTICS")
    print("="*120)
    
    total_graphs = len(df)
    total_feasible = df['feasible'].sum()
    overall_feas_rate = total_feasible / total_graphs * 100
    
    feasible_df = df[df['feasible'] == True]
    if len(feasible_df) > 0:
        overall_avg_ratio = feasible_df['approx_ratio'].mean()
    else:
        overall_avg_ratio = None
    
    print(f"Total graphs tested: {total_graphs}")
    print(f"Total feasible solutions: {total_feasible} ({overall_feas_rate:.1f}%)")
    print(f"Overall avg approx ratio: {overall_avg_ratio:.4f}" if overall_avg_ratio else "Overall avg approx ratio: N/A")
    print(f"Avg solver time: {df['time_s'].mean():.4f}s")
    print(f"Min solver time: {df['time_s'].min():.4f}s")
    print(f"Max solver time: {df['time_s'].max():.4f}s")
    
    # Overall summary table
    print(f"\n{'─'*120}")
    print(f"ALL SOLVERS/HEURISTICS - OVERALL PERFORMANCE")
    print(f"{'─'*120}")
    print(f"{'Solver/Heuristic':<40} {'Total':<8} {'Feasible':<12} {'Feas. %':<10} {'Avg Ratio':<12} {'Avg Time':<10}")
    print("-"*120)
    
    for solver in all_solvers:
        df_s = df[df['solver'] == solver]
        total_s = len(df_s)
        feasible_s = df_s['feasible'].sum()
        feas_pct = (feasible_s / total_s * 100)
        
        feasible_df_s = df_s[df_s['feasible'] == True]
        if len(feasible_df_s) > 0:
            avg_ratio_s = feasible_df_s['approx_ratio'].mean()
        else:
            avg_ratio_s = None
        
        avg_time_s = df_s['time_s'].mean()
        
        ratio_str = f"{avg_ratio_s:.4f}" if avg_ratio_s else "N/A"
        time_str = f"{avg_time_s:.4f}s"
        
        print(f"{solver:<40} {total_s:<8} {feasible_s}/{total_s:<5} {feas_pct:>9.1f}% {ratio_str:<12} {time_str:<10}")
    
    print("="*120)


def find_latest_file(pattern):
    """Find the latest file matching the pattern."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir
    files = [f for f in os.listdir(results_dir) if f.startswith(pattern) and f.endswith(".csv")]
    
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        return os.path.join(results_dir, files[0])
    return None


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    quantum_file = find_latest_file("quantum_experiment_")
    heuristics_file = find_latest_file("heuristics_experiment_")
    
    analyze_results(quantum_file, heuristics_file)
