import pandas as pd

# Read the statistics.csv file
df = pd.read_csv('statistics.csv')

# Filter for N <= 15
# The 'n' column represents the number of nodes
filtered_df = df[(df['n'] <= 15)]

print(f"Total graphs in dataset: {len(df)}")
print(f"Graphs with N <= 15: {len(filtered_df)}")
print("\nFiltered dataset:")
print(filtered_df[['name', 'n', 'm', 'category', 'bdegdel']].to_string())
