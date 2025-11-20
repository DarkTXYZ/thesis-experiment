import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import csv
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from Utils.min_lin_arrangement import calculate_min_linear_arrangement


def build_model(graph: nx.Graph, time_limit: float) -> gp.Model:
    model = gp.Model("MINLA_Direct")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    n = graph.number_of_nodes()
    edges = list(graph.edges())

    p = model.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name="p")
    d = model.addVars(len(edges), lb=0, ub=n - 1, vtype=GRB.INTEGER, name="d")
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(p[i] == gp.quicksum((j + 1) * x[i, j] for j in range(n)))
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)

    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)

    for i, (u, v) in enumerate(edges):
        model.addConstr(d[i] >= p[u] - p[v])
        model.addConstr(d[i] >= p[v] - p[u])

    model.setObjective(gp.quicksum(d[i] for i in range(len(edges))), GRB.MINIMIZE)

    return model, p, n, edges


def extract_solution(model: gp.Model, p, n: int, graph: nx.Graph) -> Tuple[List[int], float]:
    if model.SolCount == 0:
        return [], float("inf")

    ordering = [int(p[i].X) for i in range(n)]

    if None in ordering or len(set(ordering)) != n:
        print(f"WARNING: Invalid ordering detected")
        return list(range(1, n + 1)), float("inf")

    cost = float(model.ObjVal)
    verified_cost = calculate_min_linear_arrangement(graph, ordering)

    if verified_cost != cost:
        print(f"WARNING: Cost mismatch! Gurobi: {cost}, Verified: {verified_cost}")
        cost = verified_cost
    else:
        print(f"✓ Solution validated: cost={cost}")

    return ordering, cost


def get_stats(model: gp.Model) -> Dict:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED"
    }

    return {
        "status": status_map.get(model.Status, str(model.Status)),
        "solve_time": float(model.Runtime) if hasattr(model, "Runtime") else None,
        "mip_gap": float(model.MIPGap) if hasattr(model, "MIPGap") else None,
        "node_count": int(model.NodeCount) if hasattr(model, "NodeCount") else None,
        "sol_count": int(model.SolCount) if hasattr(model, "SolCount") else 0
    }


def solve_minla_direct(graph: nx.Graph, time_limit: float = 300) -> Tuple[List[int], float, Dict]:
    model, p, n, edges = build_model(graph, time_limit)
    model.optimize()
    ordering, cost = extract_solution(model, p, n, graph)
    stats = get_stats(model)
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
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
    return G


def run_testcase(time_limit: int):
    print("Using small testcase with direct ILP formulation...")
    print("=" * 80)

    G = create_testcase()
    n, m = G.number_of_nodes(), G.number_of_edges()

    print(f"\nTest Graph:")
    print(f"  Nodes: {n}, Edges: {m}")
    print(f"  Edges: {list(G.edges())}")

    ordering, cost, stats = solve_minla_direct(G, time_limit=time_limit)

    print(f"\n  MINLA Cost: {cost}")
    print(f"  Ordering: {ordering}")
    print(f"  Solve Time: {stats['solve_time']:.2f}s")
    print(f"  Status: {stats['status']}")
    print(f"  MIP Gap: {stats['mip_gap']}")


def process_graph_files(dataset_dir: Path, time_limit: int) -> List[Dict]:
    graph_files = sorted(dataset_dir.glob("*_preprocessed.txt"))

    if not graph_files:
        print(f"No preprocessed graph files found in {dataset_dir}")
        return []

    print(f"Processing {len(graph_files)} graph files with Gurobi Direct ILP solver...")
    print("=" * 80)

    results = []

    for graph_file in graph_files:
        print(f"\nProcessing: {graph_file.name}")

        try:
            G = load_graph(graph_file)
            n, m = G.number_of_nodes(), G.number_of_edges()

            print(f"  Nodes: {n}, Edges: {m}")

            ordering, cost, stats = solve_minla_direct(G, time_limit=time_limit)

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
                'node_count': stats['node_count']
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
                'node_count': 'N/A'
            })

    return results


def save_results(results: List[Dict], output_file: Path):
    print("\n" + "=" * 80)
    print(f"Writing results to {output_file}")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['graph', 'n', 'm', 'cost', 'solve_time', 'status', 'mip_gap', 'node_count']
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

    results_dir = Path(__file__).parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"gurobi_direct_results_{timestamp}.csv"

    if USE_TESTCASE:
        run_testcase(TIME_LIMIT)
    else:
        dataset_dir = Path(__file__).parent.parent / "Dataset" / "processed"
        results = process_graph_files(dataset_dir, TIME_LIMIT)
        if results:
            save_results(results, output_file)