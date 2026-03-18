import os
import pickle
import Utils.MinLA as minla
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

DATASET_PATH = "Dataset/quantum_dataset"

def read_dataset():
    # read all pickle files in DATASET_PATH
    datasets = {}
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".pkl"):
            filepath = os.path.join(DATASET_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                datasets[data['num_vertices']] = data

    return datasets


graph = read_dataset()[5]['graphs'][0]
G = nx.Graph()
G.add_nodes_from(range(graph['num_vertices']))
G.add_edges_from(graph['edges'])
n = G.number_of_nodes()

bqm = minla.generate_bqm_instance(G)
bqm.normalize()

# Display BQM as a heatmap 
# Create mapping from variable names to indices
var_to_idx = {}
idx_to_var = {}
idx = 0
for var in sorted(set(list(bqm.linear.keys()) + [v for edge in bqm.quadratic for v in edge])):
    var_to_idx[var] = idx
    idx_to_var[idx] = var
    idx += 1

num_vars = len(var_to_idx)
matrix = np.zeros((num_vars, num_vars))

# Add linear terms (diagonal)
for var, coeff in bqm.linear.items():
    i = var_to_idx[var]
    matrix[i, i] = coeff

# Add quadratic terms (upper and lower triangular)
for (var1, var2), coeff in bqm.quadratic.items():
    i = var_to_idx[var1]
    j = var_to_idx[var2]
    matrix[i, j] = coeff
    matrix[j, i] = coeff

# Create heatmap with red-blue color scheme
# Red for negative values, Blue for positive values
plt.figure(figsize=(12, 10))
sns.heatmap(matrix, cmap='RdBu_r', annot=False, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Coefficient'},
            center=0)
plt.title(f'BQM Heatmap for Graph with {n} vertices ({num_vars} variables)')
plt.xlabel('Variable Index')
plt.ylabel('Variable Index')
plt.tight_layout()
plt.show()
