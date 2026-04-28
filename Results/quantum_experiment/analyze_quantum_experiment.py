import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def analyze_results(file_path):
    """Analyze quantum experiment results from CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return None
    
    df = pd.read_csv(file_path)
    print(f"\nLoaded {len(df)} results from {file_path}")
    print(f"{'='*80}\n")
    
    print("ANALYSIS BY SAMPLER AND GRAPH SIZE (n)")
    print("="*80)
    
    all_summaries = []
    
    for sampler in sorted(df['solver'].unique()):
        df_sampler = df[df['solver'] == sampler]
        print(f"\n{sampler} ({len(df_sampler)} results)")
        print("-"*80)
        print(f"{'n':<5} {'Graphs':<8} {'Feasible':<12} {'Feas. Rate':<12} {'Avg Ratio':<16} {'Std Dev':<12} {'Avg Time':<10}")
        print("-"*80)
        
        sampler_summary = []
        
        for n in sorted(df_sampler['n'].unique()):
            df_n = df_sampler[df_sampler['n'] == n]
            num_graphs = len(df_n)
            num_feasible = df_n['feasible'].sum()
            feasibility_rate = num_feasible / num_graphs * 100
            
            feasible_df = df_n[df_n['feasible'] == True]
            if len(feasible_df) > 0:
                avg_ratio = feasible_df['approx_ratio'].mean()
                std_ratio = feasible_df['approx_ratio'].std()
            else:
                avg_ratio = None
                std_ratio = None
            
            avg_time = df_n['time_s'].mean()
            
            ratio_str = f"{avg_ratio:.4f}" if avg_ratio is not None else "N/A"
            std_str = f"{std_ratio:.4f}" if std_ratio is not None else "N/A"
            time_str = f"{avg_time:.3f}s"
            
            print(f"{n:<5} {num_graphs:<8} {num_feasible}/{num_graphs:<6} {feasibility_rate:>10.1f}% {ratio_str:<16} {std_str:<12} {time_str:<10}")
            
            sampler_summary.append({
                'sampler': sampler,
                'n': n,
                'num_graphs': num_graphs,
                'num_feasible': num_feasible,
                'feasibility_rate': feasibility_rate,
                'avg_approx_ratio': avg_ratio,
                'std_approx_ratio': std_ratio,
                'avg_time': avg_time,
            })
        
        all_summaries.extend(sampler_summary)
        
        print(f"\n{sampler} Overall:")
        print("-"*80)
        total_graphs = len(df_sampler)
        total_feasible = df_sampler['feasible'].sum()
        overall_feas_rate = total_feasible / total_graphs * 100
        
        feasible_df = df_sampler[df_sampler['feasible'] == True]
        if len(feasible_df) > 0:
            overall_avg_ratio = feasible_df['approx_ratio'].mean()
            overall_std_ratio = feasible_df['approx_ratio'].std()
        else:
            overall_avg_ratio = None
            overall_std_ratio = None
        
        print(f"Total graphs: {total_graphs}")
        print(f"Total feasible: {total_feasible} ({overall_feas_rate:.1f}%)")
        print(f"Avg approx ratio: {overall_avg_ratio:.4f}" if overall_avg_ratio else "Avg approx ratio: N/A")
        print(f"Std dev approx ratio: {overall_std_ratio:.4f}" if overall_std_ratio else "Std dev approx ratio: N/A")
        print(f"Avg solver time: {df_sampler['time_s'].mean():.3f}s")
    
    print("\n" + "="*80)
    print("\nOVERALL STATISTICS (All Samplers)")
    print("="*80)
    
    total_graphs = len(df)
    total_feasible = df['feasible'].sum()
    overall_feas_rate = total_feasible / total_graphs * 100
    
    feasible_df = df[df['feasible'] == True]
    if len(feasible_df) > 0:
        overall_avg_ratio = feasible_df['approx_ratio'].mean()
        overall_std_ratio = feasible_df['approx_ratio'].std()
    else:
        overall_avg_ratio = None
        overall_std_ratio = None
    
    print(f"Total graphs tested: {total_graphs}")
    print(f"Total feasible solutions: {total_feasible} ({overall_feas_rate:.1f}%)")
    print(f"Overall avg approx ratio: {overall_avg_ratio:.4f}" if overall_avg_ratio else "Overall avg approx ratio: N/A")
    print(f"Overall std dev approx ratio: {overall_std_ratio:.4f}" if overall_std_ratio else "Overall std dev approx ratio: N/A")
    
    print(f"\nAvg solver time: {df['time_s'].mean():.3f}s")
    print(f"Min solver time: {df['time_s'].min():.3f}s")
    print(f"Max solver time: {df['time_s'].max():.3f}s")
    
    print("\n" + "="*80)
    
    # Save summary to CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        output_filename = os.path.join(os.path.dirname(file_path), 'analysis_summary.csv')
        summary_df.to_csv(output_filename, index=False)
        print(f"\nSummary saved to: {output_filename}")
    
    return {
        'file_path': file_path,
        'total_graphs': total_graphs,
        'total_feasible': total_feasible,
        'feasibility_rate': overall_feas_rate,
        'avg_approx_ratio': overall_avg_ratio,
        'std_approx_ratio': overall_std_ratio,
        'avg_time': df['time_s'].mean(),
    }


if __name__ == "__main__":
    file_path = 'Results/quantum_experiment/quantum_experiment_custom_default_beta_20260419_184427.csv'
    
    if os.path.exists(file_path):
        analyze_results(file_path)
    else:
        print(f"Error: File '{file_path}' not found")
