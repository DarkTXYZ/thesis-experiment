import json
import pandas as pd
import os
from pathlib import Path

RESULTS_JSON = os.path.join(Path(__file__).parent.parent, "Results/PIM_tuning_fixed_temperature_results.json")

def load_results():
    """Load PIM tuning results from JSON"""
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['results'])

def print_result(row):
    """Pretty print a single result"""
    feasible_str = "✓ FEASIBLE" if row['feasible'] else "✗ NOT FEASIBLE"
    print(f"  ID: {row['id']}")
    print(f"  Status: {feasible_str}")
    print(f"  Graph: n={row['n']}, m={row['m']}")
    print(f"  Beta range: ({row['beta_min']}, {row['beta_max']})")
    print(f"  Space type: {row['space_type']}, Annealing: {row['annealing_type']}")
    print(f"  BQM Normalized: {row['bqm_is_normalized']}")
    print(f"  Energy: {row['energy']:.6f}")
    if row['feasible']:
        print(f"  MinLA cost: {row['minla_cost']}")
        print(f"  Optimal cost: {row['optimal_cost']}")
        if row['relative_gap'] is not None:
            print(f"  Relative gap: {row['relative_gap']:.4f}")
    print(f"  Time: {row['time_s']:.3f}s")

def get_top_feasible(n=10):
    """Get top N feasible results by minla_cost"""
    df = load_results()
    feasible_df = df[df['feasible'] == True]
    if len(feasible_df) == 0:
        print("No feasible results found")
        return None
    
    top = feasible_df.nsmallest(n, 'minla_cost')
    print(f"\nTop {n} feasible results (by MinLA cost):")
    print("-" * 100)
    for idx, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{idx}. ID={row['id']:3d} | Cost={row['minla_cost']:4d} | Gap={row['relative_gap']:6.2%} | "
              f"Beta=({row['beta_min']:.0e}, {row['beta_max']:.0e}) | {row['space_type']:9s} | "
              f"Norm={str(row['bqm_is_normalized']):5s} | Time={row['time_s']:6.2f}s")
    return top

def get_top_infeasible(n=10):
    """Get top N infeasible results by lowest energy"""
    df = load_results()
    infeasible_df = df[df['feasible'] == False]
    if len(infeasible_df) == 0:
        print("No infeasible results found")
        return None
    
    top = infeasible_df.nsmallest(n, 'energy')
    print(f"\nTop {n} infeasible results (by lowest energy):")
    print("-" * 100)
    for idx, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{idx}. ID={row['id']:3d} | Energy={row['energy']:12.6f} | "
              f"Beta=({row['beta_min']:.0e}, {row['beta_max']:.0e}) | {row['space_type']:9s} | "
              f"Annealing={row['annealing_type']:9s} | Norm={str(row['bqm_is_normalized']):5s} | Time={row['time_s']:6.2f}s")
    return top

def compare_normalized():
    """Compare lowest energy results grouped by normalization status (regardless of feasibility)"""
    df = load_results()
    
    print("\n" + "=" * 100)
    print(" COMPARISON: NORMALIZED vs NOT NORMALIZED (Lowest Energy)")
    print("=" * 100)
    
    for norm_status in [True, False]:
        status_str = "NORMALIZED" if norm_status else "NOT NORMALIZED"
        norm_df = df[df['bqm_is_normalized'] == norm_status]
        
        if len(norm_df) == 0:
            print(f"\n{status_str}: No results found")
            continue
        
        print(f"\n{status_str}:")
        print("-" * 100)
        print(f"  Total results: {len(norm_df)}")
        
        # Get lowest energy regardless of feasibility
        best = norm_df.nsmallest(1, 'energy').iloc[0]
        print(f"\n  Lowest Energy Result:")
        print(f"    ID: {best['id']}")
        print(f"    Energy: {best['energy']:.6f}")
        print(f"    Feasible: {best['feasible']}")
        print(f"    Graph: n={best['n']}, m={best['m']}")
        print(f"    Beta range: ({best['beta_min']}, {best['beta_max']})")
        print(f"    Space type: {best['space_type']}")
        print(f"    Annealing: {best['annealing_type']}")
        print(f"    Time: {best['time_s']:.3f}s")
        
        # Show top 5 lowest energy results
        print(f"\n  Top 5 lowest energy results:")
        top5 = norm_df.nsmallest(5, 'energy')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            feasible_str = "✓" if row['feasible'] else "✗"
            print(f"    {idx}. Energy={row['energy']:12.6f} | {feasible_str} | "
                  f"Beta=({row['beta_min']:.0e}, {row['beta_max']:.0e}) | {row['space_type']:9s} | "
                  f"{row['annealing_type']:9s}")
    
    # Show the comparison
    normalized_df = df[df['bqm_is_normalized'] == True]
    not_normalized_df = df[df['bqm_is_normalized'] == False]
    
    best_norm = normalized_df.nsmallest(1, 'energy').iloc[0]['energy']
    best_not_norm = not_normalized_df.nsmallest(1, 'energy').iloc[0]['energy']
    
    print("\n" + "-" * 100)
    print(" SUMMARY")
    print("-" * 100)
    print(f"  Normalized best energy: {best_norm:.6f}")
    print(f"  Not Normalized best energy: {best_not_norm:.6f}")
    print(f"  Improvement factor: {best_not_norm / best_norm:.1f}x (normalized is better)")
    print("=" * 100)

def get_top_10_by_normalized(n=10):
    """Get top N results grouped by normalization status (feasible or not)"""
    df = load_results()
    
    print("\n" + "=" * 150)
    print(" TOP 10 RESULTS GROUPED BY NORMALIZED STATUS (Feasible & Infeasible)")
    print("=" * 150)
    
    for norm_status in [True, False]:
        status_str = "NORMALIZED" if norm_status else "NOT NORMALIZED"
        norm_df = df[df['bqm_is_normalized'] == norm_status]
        
        if len(norm_df) == 0:
            print(f"\n{status_str}: No results found")
            continue
        
        # Sort by energy (lowest first)
        top_n = norm_df.nsmallest(n, 'energy')
        
        print(f"\n{status_str}: ({len(norm_df)} total results)")
        print("-" * 150)
        print(f"{'Rank':<5} {'ID':<5} {'Energy':<14} {'Feasible':<10} {'Cost':<8} {'Gap':<8} "
              f"{'Beta Range':<30} {'Space Type':<12} {'Annealing':<12} {'Time(s)':<8}")
        print("-" * 150)
        
        for idx, (_, row) in enumerate(top_n.iterrows(), 1):
            feasible_str = "✓ YES" if row['feasible'] else "✗ NO"
            cost_str = f"{row['minla_cost']}" if row['feasible'] else "-"
            gap_str = f"{row['relative_gap']:.2%}" if (row['feasible'] and row['relative_gap'] is not None) else "-"
            beta_range = f"({row['beta_min']:.0e}, {row['beta_max']:.0e})"
            
            print(f"{idx:<5} {row['id']:<5} {row['energy']:<14.6f} {feasible_str:<10} {cost_str:<8} "
                  f"{gap_str:<8} {beta_range:<30} {row['space_type']:<12} {row['annealing_type']:<12} {row['time_s']:<8.3f}")
    
    print("\n" + "=" * 150)

if __name__ == "__main__":
    get_top_feasible(10)
    print("\n")
    get_top_infeasible(10)
    print("\n")
    get_top_10_by_normalized(10)
