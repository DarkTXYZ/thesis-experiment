import glob
import os
import pickle
import sys

import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from Utils.MinLA import find_all_minimum_solutions

DATASET_DIR = SCRIPT_DIR
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "exact_dataset_n5.pkl")


def main():
    results = {}
    
    # Process graphs from n3, n4, and n5 subdirectories
    for num_nodes in [3, 4, 5]:
        subdir = os.path.join(DATASET_DIR, f"n{num_nodes}")
        if not os.path.exists(subdir):
            print(f"Directory {subdir} not found, skipping...")
            continue
            
        edgelist_files = sorted(glob.glob(os.path.join(subdir, "*.edgelist")))
        print(f"\nProcessing {len(edgelist_files)} graphs from n{num_nodes}...")

        for filepath in edgelist_files:
            graph_name = os.path.splitext(os.path.basename(filepath))[0]
            graph = nx.read_edgelist(filepath, nodetype=int)

            status, solutions = find_all_minimum_solutions(graph)

            # Compute objective value from first solution (all are optimal)
            objective_value = None
            if solutions:
                pos = {node: label for node, label in enumerate(solutions[0])}
                objective_value = sum(abs(pos[u] - pos[v]) for u, v in graph.edges())

            results[f"n{num_nodes}_{graph_name}"] = {
                "status": status,
                "objective_value": objective_value,
                "num_solutions": len(solutions),
                "solutions": solutions,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            }
            print(f"  {graph_name}: obj={objective_value}, #solutions={len(solutions)}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
