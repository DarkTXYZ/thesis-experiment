"""
Hyperparameter tuning experiment for openjij SQA on a 30-vertex graph.

This script runs a configurable grid search over SQA parameters using one graph
from Dataset/quantum_dataset/quantum_n30.pkl and reports the best settings.
"""

import os
import sys
import pickle
import time
import itertools
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import openjij as oj


# Add parent directory to path to import Utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Utils.MinLA as minla


# Paths
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PARENT_DIR, "Dataset/quantum_dataset")
RESULTS_DIR = os.path.join(PARENT_DIR, "Results")


# Experiment configuration
SEED = 42
NUM_VERTICES = 30
GRAPH_INDEX = 0
NUM_READS = 10
NUM_SWEEPS = 1000
SAVE_CSV = True


# Hyperparameter search space
BETA_MIN_LIST = [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1, 5, 10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
# BETA_MAX_LIST = [1.0, 5.0, 10.0, 50.0]
GAMMA_LIST = [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1, 5, 10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
TROTTER_LIST = [8]
SPARSE_LIST = [True]
REINITIALIZE_LIST = [True]

def read_dataset_for_vertices(num_vertices: int):
	"""Load dataset pickle file for a specific vertex count."""
	filepath = os.path.join(DATASET_PATH, f"quantum_n{num_vertices}.pkl")
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"Dataset not found: {filepath}")
	with open(filepath, "rb") as f:
		return pickle.load(f)


def convert_graph_data_to_nx(graph_data):
	"""Convert dataset graph dictionary to a NetworkX graph."""
	graph_nx = nx.Graph()
	graph_nx.add_nodes_from(range(graph_data["num_vertices"]))
	graph_nx.add_edges_from(graph_data["edges"])
	return graph_nx


def check_feasibility(sol: np.ndarray, n: int) -> bool:
	"""Check if decoded solution satisfies monotonicity and full label coverage."""
	for u in range(n):
		if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
			return False
	labels = set(np.sum(sol, axis=1))
	return labels == set(range(1, n + 1))


def decode_solution(raw_sample: Dict, n: int) -> Tuple[np.ndarray, bool]:
	"""Decode X[u][k] variables into ordering and feasibility."""
	sol = np.array(
		[[raw_sample.get(f"X[{u}][{k}]", 0) for k in range(n)] for u in range(n)],
		dtype=int,
	)
	is_feasible = check_feasibility(sol, n)
	ordering = np.sum(sol, axis=1)
	return ordering, is_feasible


def evaluate_configuration(
	bqm,
	graph_nx: nx.Graph,
	n: int,
	optimal_cost,
	beta_min: float,
	gamma: float,
	trotter: int,
	sparse: bool,
	reinitialize_state: bool,
):
	"""Run one SQA configuration and return metrics for reporting."""
	solver = oj.SQASampler()

	start = time.time()
	sampleset = solver.sample(
		bqm,
		num_reads=NUM_READS,
		num_sweeps=NUM_SWEEPS,
		beta=beta_min,
		gamma=gamma,
		trotter=trotter,
		sparse=sparse,
		reinitialize_state=reinitialize_state,
		seed=SEED,
	)
	elapsed = time.time() - start

	best = sampleset.first
	energy = best.energy
	ordering, feasible = decode_solution(best.sample, n)
	minla_cost = minla.calculate_min_linear_arrangement(graph_nx, ordering) if feasible else None
	relative_gap = ((minla_cost - optimal_cost) / optimal_cost) if (feasible and optimal_cost) else None

	return {
		"energy": energy,
		"feasible": feasible,
		"minla_cost": minla_cost,
		"relative_gap": relative_gap,
		"time_s": round(elapsed, 4),
	}


def run_experiment():
	"""Run OpenJij SQA hyperparameter tuning for one 30-vertex graph."""
	np.random.seed(SEED)

	print(f"Loading quantum dataset with n={NUM_VERTICES}...")
	dataset = read_dataset_for_vertices(NUM_VERTICES)
	graphs = dataset["graphs"]

	if GRAPH_INDEX >= len(graphs):
		raise IndexError(f"GRAPH_INDEX={GRAPH_INDEX} out of range for {len(graphs)} graphs")

	graph_data = graphs[GRAPH_INDEX]
	graph_nx = convert_graph_data_to_nx(graph_data)
	n = graph_nx.number_of_nodes()
	m = graph_nx.number_of_edges()
	optimal_cost = graph_data.get("optimal_cost", None)

	bqm = minla.generate_bqm_instance(graph_nx)
	bqm.normalize()

	print(f"Graph index: {GRAPH_INDEX} | n={n}, m={m}, optimal_cost={optimal_cost}")
	print(f"num_reads={NUM_READS}, num_sweeps={NUM_SWEEPS}, seed={SEED}")

	combinations = [
		combo
		for combo in itertools.product(
			BETA_MIN_LIST,
			GAMMA_LIST,
			TROTTER_LIST,
			SPARSE_LIST,
			REINITIALIZE_LIST,
		)
	]

	total = len(combinations)
	print(f"Total valid configurations: {total}\n")

	rows = []
	for idx, (beta_min, gamma, trotter, sparse, reinit) in enumerate(combinations, start=1):
		try:
			result = evaluate_configuration(
				bqm=bqm,
				graph_nx=graph_nx,
				n=n,
				optimal_cost=optimal_cost,
				beta_min=beta_min,
				gamma=gamma,
				trotter=trotter,
				sparse=sparse,
				reinitialize_state=reinit,
			)
			status = "OK"
			error_msg = None
		except Exception as exc:
			result = {
				"energy": None,
				"feasible": False,
				"minla_cost": None,
				"relative_gap": None,
				"time_s": None,
			}
			status = "ERR"
			error_msg = str(exc)

		row = {
			"config_id": idx,
			"graph_index": GRAPH_INDEX,
			"n": n,
			"m": m,
			"num_reads": NUM_READS,
			"num_sweeps": NUM_SWEEPS,
			"beta_min": beta_min,
			"gamma": gamma,
			"trotter": trotter,
			"sparse": sparse,
			"reinitialize_state": reinit,
			"energy": result["energy"],
			"feasible": result["feasible"],
			"minla_cost": result["minla_cost"],
			"optimal_cost": optimal_cost,
			"relative_gap": result["relative_gap"],
			"time_s": result["time_s"],
			"status": status,
			"error": error_msg,
		}
		rows.append(row)

		if status == "OK":
			print(
				f"[{idx:3d}/{total}] {status} | "
				f"beta=({beta_min:.1e}) | gamma={gamma:.2f} | trotter={trotter} | "
				f"E={result['energy']:.2f} | feas={result['feasible']} | t={result['time_s']:.3f}s"
			)
		else:
			print(
				f"[{idx:3d}/{total}] {status} | "
				f"beta=({beta_min:.1e}) | gamma={gamma:.2f} | trotter={trotter} | "
				f"error={error_msg}"
			)

	df = pd.DataFrame(rows)

	ok_df = df[df["status"] == "OK"].copy()
	if not ok_df.empty:
		ranked = ok_df.sort_values(
			by=["feasible", "energy", "time_s"],
			ascending=[False, True, True],
		)

		print("\n" + "=" * 80)
		print("TOP 10 CONFIGURATIONS (feasible first, then lower energy/time)")
		print("=" * 80)
		print(
			ranked[
				[
					"config_id",
					"beta_min",
					"gamma",
					"trotter",
					"energy",
					"feasible",
					"minla_cost",
					"relative_gap",
					"time_s",
				]
			]
			.head(10)
			.to_string(index=False)
		)

	print("\n" + "=" * 80)
	print("SUMMARY")
	print("=" * 80)
	print(f"Total configs: {len(df)}")
	print(f"Successful runs: {(df['status'] == 'OK').sum()}")
	print(f"Failed runs: {(df['status'] == 'ERR').sum()}")
	print(f"Feasible solutions: {df['feasible'].sum()}")

	if SAVE_CSV:
		os.makedirs(RESULTS_DIR, exist_ok=True)
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		csv_path = os.path.join(RESULTS_DIR, f"openjij_hypertuning_n{NUM_VERTICES}_g{GRAPH_INDEX}_{timestamp}.csv")
		df.to_csv(csv_path, index=False)
		print(f"Results saved to {csv_path}")

	return df


if __name__ == "__main__":
	run_experiment()
