"""
Compare QWaveSampler results from three annealing schedule files:
- default.csv: Default annealing schedule
- linear.csv: Linear annealing schedule
- expo.csv: Exponential annealing schedule
"""

import pandas as pd

# Read CSV files
df_default = pd.read_csv('default_schedule.csv')
df_linear = pd.read_csv('linear_schedule.csv')
df_expo = pd.read_csv('expo_schedule.csv')

# Filter only QWaveSampler results
df_default = df_default[df_default['solver_name'].str.contains('QWaveSampler')].copy()
df_linear = df_linear[df_linear['solver_name'].str.contains('QWaveSampler')].copy()
df_expo = df_expo[df_expo['solver_name'].str.contains('QWaveSampler')].copy()

print("=" * 100)
print("COMPARISON: Default vs Linear vs Exponential Annealing Schedules - QWaveSampler")
print("=" * 100)

# Merge on common keys
merge_keys = ['dataset_name', 'num_vertices', 'penalty_mode']
df_compare = pd.merge(df_default, df_linear, on=merge_keys, suffixes=('_default', '_linear'))
df_compare = pd.merge(df_compare, df_expo, on=merge_keys)
# Rename expo columns
expo_cols = ['feasibility_rate', 'success_rate', 'avg_relative_gap', 'total_time', 'num_feasible', 'num_success']
for col in expo_cols:
    if col in df_compare.columns:
        df_compare = df_compare.rename(columns={col: f'{col}_expo'})

# Select key metrics to compare
print("\n" + "=" * 100)
print("FEASIBILITY RATE COMPARISON")
print("=" * 100)
print(f"{'Dataset':20} | {'Penalty':6} | {'Default':>10} | {'Linear':>10} | {'Expo':>10} | {'Best':>15}")
print("-" * 100)
for _, row in df_compare.iterrows():
    dataset = row['dataset_name']
    penalty = row['penalty_mode']
    feas_def = row['feasibility_rate_default']
    feas_lin = row['feasibility_rate_linear']
    feas_exp = row['feasibility_rate_expo']
    best = max(feas_def, feas_lin, feas_exp)
    best_names = []
    if feas_def == best: best_names.append('Default')
    if feas_lin == best: best_names.append('Linear')
    if feas_exp == best: best_names.append('Expo')
    best_str = ', '.join(best_names)
    print(f"{dataset:20} | {penalty:6} | {feas_def:>10.2%} | {feas_lin:>10.2%} | {feas_exp:>10.2%} | {best_str:>15}")

print("\n" + "=" * 100)
print("SUCCESS RATE COMPARISON")
print("=" * 100)
print(f"{'Dataset':20} | {'Penalty':6} | {'Default':>10} | {'Linear':>10} | {'Expo':>10} | {'Best':>15}")
print("-" * 100)
for _, row in df_compare.iterrows():
    dataset = row['dataset_name']
    penalty = row['penalty_mode']
    succ_def = row['success_rate_default']
    succ_lin = row['success_rate_linear']
    succ_exp = row['success_rate_expo']
    best = max(succ_def, succ_lin, succ_exp)
    best_names = []
    if succ_def == best: best_names.append('Default')
    if succ_lin == best: best_names.append('Linear')
    if succ_exp == best: best_names.append('Expo')
    best_str = ', '.join(best_names)
    print(f"{dataset:20} | {penalty:6} | {succ_def:>10.2%} | {succ_lin:>10.2%} | {succ_exp:>10.2%} | {best_str:>15}")

print("\n" + "=" * 100)
print("AVG RELATIVE GAP COMPARISON (lower is better)")
print("=" * 100)
print(f"{'Dataset':20} | {'Penalty':6} | {'Default':>10} | {'Linear':>10} | {'Expo':>10} | {'Best':>15}")
print("-" * 100)
for _, row in df_compare.iterrows():
    dataset = row['dataset_name']
    penalty = row['penalty_mode']
    gap_def = row['avg_relative_gap_default']
    gap_lin = row['avg_relative_gap_linear']
    gap_exp = row['avg_relative_gap_expo']
    
    # Handle inf values
    def fmt_gap(g):
        return f"{g:.2%}" if g != float('inf') else "inf"
    
    # Find best (lowest non-inf)
    gaps = [(gap_def, 'Default'), (gap_lin, 'Linear'), (gap_exp, 'Expo')]
    valid_gaps = [(g, n) for g, n in gaps if g != float('inf')]
    if valid_gaps:
        best_gap = min(g for g, n in valid_gaps)
        best_names = [n for g, n in valid_gaps if g == best_gap]
        best_str = ', '.join(best_names)
    else:
        best_str = 'N/A'
    
    print(f"{dataset:20} | {penalty:6} | {fmt_gap(gap_def):>10} | {fmt_gap(gap_lin):>10} | {fmt_gap(gap_exp):>10} | {best_str:>15}")

print("\n" + "=" * 100)
print("TOTAL TIME COMPARISON (seconds)")
print("=" * 100)
print(f"{'Dataset':20} | {'Penalty':6} | {'Default':>10} | {'Linear':>10} | {'Expo':>10} | {'Fastest':>15}")
print("-" * 100)
for _, row in df_compare.iterrows():
    dataset = row['dataset_name']
    penalty = row['penalty_mode']
    time_def = row['total_time_default']
    time_lin = row['total_time_linear']
    time_exp = row['total_time_expo']
    fastest = min(time_def, time_lin, time_exp)
    fastest_names = []
    if time_def == fastest: fastest_names.append('Default')
    if time_lin == fastest: fastest_names.append('Linear')
    if time_exp == fastest: fastest_names.append('Expo')
    fastest_str = ', '.join(fastest_names)
    print(f"{dataset:20} | {penalty:6} | {time_def:>9.1f}s | {time_lin:>9.1f}s | {time_exp:>9.1f}s | {fastest_str:>15}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

# Calculate averages
avg_feas = {
    'Default': df_compare['feasibility_rate_default'].mean(),
    'Linear': df_compare['feasibility_rate_linear'].mean(),
    'Expo': df_compare['feasibility_rate_expo'].mean()
}
avg_succ = {
    'Default': df_compare['success_rate_default'].mean(),
    'Linear': df_compare['success_rate_linear'].mean(),
    'Expo': df_compare['success_rate_expo'].mean()
}

# Filter out inf values for gap comparison
def get_valid_mean(series):
    valid = series[series != float('inf')]
    return valid.mean() if len(valid) > 0 else float('nan')

avg_gap = {
    'Default': get_valid_mean(df_compare['avg_relative_gap_default']),
    'Linear': get_valid_mean(df_compare['avg_relative_gap_linear']),
    'Expo': get_valid_mean(df_compare['avg_relative_gap_expo'])
}

total_time = {
    'Default': df_compare['total_time_default'].sum(),
    'Linear': df_compare['total_time_linear'].sum(),
    'Expo': df_compare['total_time_expo'].sum()
}

print(f"\n{'Metric':<25} | {'Default':>12} | {'Linear':>12} | {'Expo':>12}")
print("-" * 70)
print(f"{'Avg Feasibility Rate':<25} | {avg_feas['Default']:>12.2%} | {avg_feas['Linear']:>12.2%} | {avg_feas['Expo']:>12.2%}")
print(f"{'Avg Success Rate':<25} | {avg_succ['Default']:>12.2%} | {avg_succ['Linear']:>12.2%} | {avg_succ['Expo']:>12.2%}")
print(f"{'Avg Relative Gap':<25} | {avg_gap['Default']:>12.2%} | {avg_gap['Linear']:>12.2%} | {avg_gap['Expo']:>12.2%}")
print(f"{'Total Time (s)':<25} | {total_time['Default']:>12.1f} | {total_time['Linear']:>12.1f} | {total_time['Expo']:>12.1f}")