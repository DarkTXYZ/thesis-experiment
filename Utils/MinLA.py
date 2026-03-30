import dimod
import networkx as nx
from pyqubo import Array
from typing import List
from itertools import permutations
from ortools.sat.python import cp_model 
import gurobipy as gp
from gurobipy import GRB

def calculate_upper_obj_bound(G: nx.Graph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    naive = m * (n - 1)
    complete = n * (n - 1) * (n + 1) / 6.0
    
    number_of_edges_remaining = m
    edges = 0
    edge_length = n - 1
    while number_of_edges_remaining > 0:
        possible_edges_at_length = n - edge_length
        if (number_of_edges_remaining >= possible_edges_at_length):
            edges += possible_edges_at_length * edge_length
            number_of_edges_remaining -= possible_edges_at_length
            edge_length -= 1
        else:
            edges += number_of_edges_remaining * edge_length
            number_of_edges_remaining = 0
            edge_length -= 1
    
    return int(min(naive, complete, edges)) + 1

def calculate_lower_obj_bound(G: nx.Graph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    number_of_edges_remaining = m
    edges = 0
    edge_length = 1
    while number_of_edges_remaining > 0:
        possible_edges_at_length = n - edge_length
        if (number_of_edges_remaining >= possible_edges_at_length):
            edges += possible_edges_at_length * edge_length
            number_of_edges_remaining -= possible_edges_at_length
            edge_length += 1
        else:
            edges += number_of_edges_remaining *  edge_length
            number_of_edges_remaining = 0
            edge_length += 1
    
    
    degree = 0
    for u in G.nodes():
        d = G.degree(u)
        if d % 2 == 0:
            degree += d*d/4 + d/2
        else:
            degree += (d*d + 2*d + 1)/4
    
    degree = degree // 2
    print(f"\tLower Bound Calculation: edges={edges}, degree={degree}")
    
    return int(max(edges, degree)) + 1

def calculate_exact_bound(G: nx.Graph):
    upper_bound_obj = calculate_upper_obj_bound(G)
    return upper_bound_obj + 1, upper_bound_obj + 1

def calculate_lucas_bound(G: nx.Graph):
    degree_sequence = (d for _, d in G.degree())
    delta = max(degree_sequence)
    
    mu_thermometer = delta + 1
    mu_bijective = delta + 1
    
    return mu_thermometer, mu_bijective

def get_penalties(graph: nx.Graph, penalty_mode: str = 'lucas') -> tuple[float, float]:
    if penalty_mode == "lucas":
        return calculate_lucas_bound(graph)
    elif penalty_mode == "exact":
        return calculate_exact_bound(graph)
    raise ValueError(f"Unknown penalty mode: {penalty_mode}")

def generate_bqm_instance(graph: nx.Graph) -> dimod.BinaryQuadraticModel:
    n = graph.number_of_nodes()
    mu_thermo, mu_bijec = get_penalties(graph)
    
    X = Array.create('X', shape=(n, n), vartype='BINARY')
    
    H_thermometer = mu_thermo * sum(
        (1 - X[u][k]) * X[u][k+1]
        for u in range(n)
        for k in range(n-1)
    )
    
    H_bijective = mu_bijec * sum(
        ((n - k) - sum(X[u][k] for u in range(n))) ** 2
        for k in range(n)
    )
    
    H_objective = sum(
        sum(X[u][k] + X[v][k] - 2 * X[u][k] * X[v][k] for k in range(n))
        for u, v in graph.edges
    )
    
    H = H_thermometer + H_bijective + H_objective
    model = H.compile()
    return model.to_bqm()

def calculate_min_linear_arrangement(graph: nx.Graph, ordering: List[int]):
    position = {vertex: label for vertex, label in enumerate(ordering)}
    cost = 0
    for u, v in graph.edges():
        cost += abs(position[u] - position[v])
    return cost

def find_one_minimum_solution(graph: nx.Graph):
    n = graph.number_of_nodes()
    
    model = cp_model.CpModel()
    
    choices = n
    node_labels = [model.new_int_var(0, choices - 1, f'X[{u}]') for u in range(n)]
    
    model.add_all_different(node_labels)
    
    objective_terms = []
    for u, v in graph.edges():
        diff = model.new_int_var(0, choices - 1, f'diff[{u}][{v}]')
        model.add_abs_equality(diff, node_labels[u] - node_labels[v])
        objective_terms.append(diff)
    
    model.minimize(sum(objective_terms))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.solve(model)
    
    obtained_label = [solver.value(node_labels[u]) for u in range(n)]
    objective_value = solver.objective_value
    
    return status.name, obtained_label, objective_value
    

class _SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._variables = variables
        self.solutions = []

    def on_solution_callback(self):
        solution = [self.Value(v) for v in self._variables]
        self.solutions.append(solution)


def find_all_minimum_solutions(graph: nx.Graph):
    n = graph.number_of_nodes()
    
    # Step 1: Find the optimal objective value
    model = cp_model.CpModel()
    choices = n
    node_labels = [model.new_int_var(0, choices - 1, f'X[{u}]') for u in range(n)]
    model.add_all_different(node_labels)
    
    objective_terms = []
    for u, v in graph.edges():
        diff = model.new_int_var(0, choices - 1, f'diff[{u}][{v}]')
        model.add_abs_equality(diff, node_labels[u] - node_labels[v])
        objective_terms.append(diff)
    
    total_obj = sum(objective_terms)
    model.minimize(total_obj)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.solve(model)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return cp_model.INFEASIBLE.name, []
    
    optimal_value = int(solver.objective_value)
    
    # Step 2: Replace objective with constraint and enumerate all solutions
    model.clear_objective()
    model.add(total_obj == optimal_value)
    
    collector = _SolutionCollector(node_labels)
    solver2 = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds = 120
    solver2.parameters.enumerate_all_solutions = True
    status2 = solver2.solve(model, collector)
    
    return status2.name, collector.solutions

def solve_minla_gurobi(G: nx.Graph):
    """
    Solves the Minimum Linear Arrangement exactly using Gurobi 
    via the Linear Ordering Polytope formulation.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Initialize the Gurobi environment and model
    # For academic licenses, create environment without empty=True first
    try:
        # Try with academic license (requires proper license setup)
        env = gp.Env()
        m = gp.Model("MinLA_LinearOrdering", env=env)
    except gp.GurobiError as e:
        print(f"Gurobi license error: {e}")
        print("Trying alternative environment setup...")
        # Fallback: create environment with empty=True
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1)
        env.start()
        m = gp.Model("MinLA_LinearOrdering", env=env)
    
    # 1. Variables: y[i, j] = 1 if node i precedes node j
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[nodes[i], nodes[j]] = m.addVar(vtype=GRB.BINARY, name=f"y_{nodes[i]}_{nodes[j]}")
            
    # Helper function to evaluate order regardless of index
    def Y(u, v):
        if u == v: return 0
        # If u < v, return y_uv. If u > v, return 1 - y_vu
        return y[u, v] if (u, v) in y else 1 - y[v, u]

    # Continuous variables for positions and distances (integrality is guaranteed by y)
    pos = {v: m.addVar(lb=1, ub=n, vtype=GRB.CONTINUOUS, name=f"pos_{v}") for v in nodes}
    dist = {e: m.addVar(lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name=f"dist_{e[0]}_{e[1]}") for e in G.edges()}
    
    # 2. Position Constraints: pos(i) = 1 + (number of nodes placed before i)
    for i in nodes:
        m.addConstr(pos[i] == 1 + gp.quicksum(Y(j, i) for j in nodes if j != i), name=f"def_pos_{i}")
        
    # 3. Distance Constraints: dist >= |pos_u - pos_v|
    for u, v in G.edges():
        m.addConstr(dist[u, v] >= pos[u] - pos[v], name=f"dist_pos_{u}_{v}")
        m.addConstr(dist[u, v] >= pos[v] - pos[u], name=f"dist_neg_{u}_{v}")
        
    # 4. Transitivity Constraints (The core of the tight LP relaxation)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                u, v, w = nodes[i], nodes[j], nodes[k]
                m.addConstr(y[u, v] + y[v, w] - y[u, w] <= 1, name=f"trans_up_{u}_{v}_{w}")
                m.addConstr(y[u, v] + y[v, w] - y[u, w] >= 0, name=f"trans_dn_{u}_{v}_{w}")
                
    # 5. Objective: Minimize total edge distances
    m.setObjective(gp.quicksum(dist[e] for e in G.edges()), GRB.MINIMIZE)
    
    # --- Solver Tuning for MinLA ---
    # Focus heavily on proving the lower bound (optimality) since heuristics find good upper bounds fast
    m.Params.MIPFocus = 2 
    # Use aggressive Gomory fractional cuts
    m.Params.GomoryPasses = 5 
    # Set a reasonable time limit (e.g., 1 hour)
    m.Params.TimeLimit = 3600 
    
    # Solve
    m.optimize()
    
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        if m.SolCount > 0:
            # Extract the 1D layout
            layout = {v: int(round(pos[v].X)) for v in nodes}
            # Sort the dictionary by position for easy reading
            sorted_layout = dict(sorted(layout.items(), key=lambda item: item[1]))
            return sorted_layout, m.ObjVal
    
    return None, None

if __name__ == "__main__":
    # Generate a random general graph with N=25
    G = nx.erdos_renyi_graph(25, 0.5, seed=42)
    
    # Ensure it's connected for a meaningful MinLA
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    print(f"Solving MinLA for N={G.number_of_nodes()}...")
    layout, cost = solve_minla_gurobi(G)
    
    print(f"\nOptimal Layout: {layout}")
    print(f"Minimum Linear Arrangement Cost: {cost}")