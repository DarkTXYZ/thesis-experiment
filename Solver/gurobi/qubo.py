import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import csv
import time
import math
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def calculate_penalty(n: int, m: int) -> int:
    naive = m * (n - 1)
    complete = n * (n - 1) * (n + 1) / 6.0
    k = math.ceil(n + 0.5 - 0.5 * math.sqrt(8.0 * m + 1))
    f = (n - k) * (n - k + 1) / 2.0
    edges_method = (m - f) * (k - 1) + (n - k) * (n * n + (n + 3) * k - 2 * k * k - 1) / 6.0
    return int(min(naive, complete, edges_method)) + 1


def var_to_index(N: int, u: int, k: int) -> int:
    return u * N + k


def build_qubo_model(graph: nx.Graph, penalty: float, time_limit: float) -> Tuple[gp.Model, Dict]:
    model = gp.Model("MINLA_QUBO")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    N = graph.number_of_nodes()
    edges = list(graph.edges())
    x = model.addVars(N * N, vtype=GRB.BINARY, name="x")
    obj = gp.QuadExpr()

    for (u, v) in edges:
        for k in range(N):
            u_ind = var_to_index(N, u, k)
            v_ind = var_to_index(N, v, k)
            obj.add(x[u_ind] + x[v_ind] - 2 * x[u_ind] * x[v_ind])

    for u in range(N):
        for k in range(N - 1):
            u_k = var_to_index(N, u, k)
            u_k1 = var_to_index(N, u, k + 1)
            obj.add((1 - x[u_k]) * x[u_k1], penalty)

    for k in range(N):
        count = sum(x[var_to_index(N, u, k)] for u in range(N))
        obj.add((count - (N - k)) ** 2, penalty)

    model.setObjective(obj, GRB.MINIMIZE)
    return model, x


def get_stats(model: gp.Model, solve_time: float) -> Dict:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED"
    }

    return {
        'status': status_map.get(model.Status, str(model.Status)),
        'solve_time': solve_time,
        'objective': model.ObjVal if model.Status == GRB.OPTIMAL else None,
        'mip_gap': float(model.MIPGap) if hasattr(model, 'MIPGap') else None,
        'node_count': int(model.NodeCount) if hasattr(model, 'NodeCount') else None
    }


def solve_qubo(graph: nx.Graph, penalty: float, time_limit: float) -> Tuple[Dict, float, Dict]:
    model, x = build_qubo_model(graph, penalty, time_limit)
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    N = graph.number_of_nodes()
    solution = {i: x[i].X for i in range(N * N)}
    stats = get_stats(model, solve_time)

    return solution, model.ObjVal, stats


def decode_thermometer(solution: Dict, n: int) -> List[int]:
    ordering = []

    for i in range(n):
        label = 0
        for k in range(n):
            idx = i * n + k
            if k < n - 1 and solution[idx] < solution[idx + 1]:
                raise ValueError(f"Invalid thermometer encoding for vertex {i}")
            label += int(solution[idx])
        ordering.append(label)

    if sorted(ordering) != list(range(1, n + 1)):
        raise ValueError("Decoded ordering is not a valid permutation")

    return ordering


def solve_minla_qubo(graph: nx.Graph, penalty: float, time_limit: float) -> Tuple[List[int], float, Dict]:
    solution, objective, stats = solve_qubo(graph, penalty, time_limit)
    ordering = decode_thermometer(solution, graph.number_of_nodes())
    cost = calculate_min_linear_arrangement(graph, ordering)

    if abs(objective - cost) > 0.01:
        print(f"WARNING: Objective mismatch! QUBO: {objective}, Calculated: {cost}")

    stats['minla_cost'] = cost
    stats['qubo_objective'] = objective

    return ordering, cost, stats


def load_graph(filepath: Path) -> nx.Graph:
    G = nx.Graph()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        n, m = map(int, lines[0].split())
        G.add_nodes_from(range(n))
        for line in lines[1:]:
            if line.strip():
                u, v = map(int, line.split())
                G.add_edge(u, v)
    return G


def create_testcase() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (4, 2)])
    return G


def run_testcase(time_limit: int):
    print("Using small testcase...")
    print("=" * 80)

    G = create_testcase()
    n, m = G.number_of_nodes(), G.number_of_edges()

    print(f"\nTest Graph:")
    print(f"  Nodes: {n}, Edges: {m}")
    print(f"  Edges: {list(G.edges())}")

    penalty = calculate_penalty(n, m)
    print(f"  Penalty: {penalty}")

    ordering, cost, stats = solve_minla_qubo(G, penalty=penalty, time_limit=time_limit)

    print(f"\n  MINLA Cost: {cost}")
    print(f"  Ordering: {ordering}")
    print(f"  Solve Time: {stats['solve_time']:.2f}s")
    print(f"  Status: {stats['status']}")
    print(f"  QUBO Objective: {stats['qubo_objective']}")


def process_graph_files(dataset_dir: Path, time_limit: int) -> List[Dict]:
    graph_files = sorted(dataset_dir.glob("*_preprocessed.txt"))

    if not graph_files:
        print(f"No preprocessed graph files found in {dataset_dir}")
        return []

    print(f"Processing {len(graph_files)} graph files with Gurobi QUBO solver...")
    print("=" * 80)

    results = []

    for graph_file in graph_files:
        print(f"\nProcessing: {graph_file.name}")

        try:
            G = load_graph(graph_file)
            n, m = G.number_of_nodes(), G.number_of_edges()

            print(f"  Nodes: {n}, Edges: {m}")

            penalty = calculate_penalty(n, m)
            print(f"  Penalty: {penalty}")

            ordering, cost, stats = solve_minla_qubo(G, penalty=penalty, time_limit=time_limit)

            print(f"  MINLA Cost: {cost}")
            print(f"  Solve Time: {stats['solve_time']:.2f}s")
            print(f"  Status: {stats['status']}")

            results.append({
                'graph': graph_file.stem.replace('_preprocessed', ''),
                'n': n,
                'm': m,
                'cost': cost,
                'solve_time': stats['solve_time'],
                'status': stats['status'],
                'mip_gap': stats['mip_gap'],
                'node_count': stats['node_count'],
                'qubo_objective': stats['qubo_objective']
            })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                'graph': graph_file.stem.replace('_preprocessed', ''),
                'n': 'N/A',
                'm': 'N/A',
                'cost': 'ERROR',
                'solve_time': 'N/A',
                'status': 'ERROR',
                'mip_gap': 'N/A',
                'node_count': 'N/A',
                'qubo_objective': 'N/A'
            })

    return results


def save_results(results: List[Dict], output_file: Path):
    print("\n" + "=" * 80)
    print(f"Writing results to {output_file}")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['graph', 'n', 'm', 'cost', 'solve_time', 'status', 'mip_gap', 'node_count', 'qubo_objective']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved successfully!")
    print("\nSummary:")
    print("-" * 80)
    for result in results:
        print(f"{result['graph']:30s} | Cost: {result['cost']:10} | Time: {result['solve_time']:>8}s")


if __name__ == "__main__":
    USE_TESTCASE = False
    TIME_LIMIT = 60

    results_dir = Path(__file__).parent.parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"gurobi_qubo_results_{timestamp}.csv"

    if USE_TESTCASE:
        run_testcase(TIME_LIMIT)
    else:
        dataset_dir = Path(__file__).parent.parent.parent / "Dataset" / "processed"
        results = process_graph_files(dataset_dir, TIME_LIMIT)
        if results:
            save_results(results, output_file)
