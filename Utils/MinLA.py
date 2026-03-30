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

def solve_MinLA_gurobi(graph: nx.Graph):
    """
    Solve the Minimum Linear Arrangement (MinLA) problem using Gurobi.
    """
    n = graph.number_of_nodes()
    
    # Extract actual node labels to prevent KeyErrors
    nodes = list(graph.nodes())
    positions = list(range(n))
    
    # Create a new Gurobi model
    model = gp.Model("MinLA")
    model.setParam('OutputFlag', 1)  # Set to 0 to suppress output
    
    # 1. Assignment Variables: x[u, p] is 1 if node u is at position p
    x = model.addVars(nodes, positions, vtype=GRB.BINARY, name="x")
    
    # Position variables (can be continuous because x dictates the integer value)
    pos = model.addVars(nodes, lb=0, ub=n-1, vtype=GRB.CONTINUOUS, name="pos")
    
    # Constraint: Each node gets exactly one position
    for u in nodes:
        model.addConstr(gp.quicksum(x[u, p] for p in positions) == 1, name=f"node_{u}_assign")
        
    # Constraint: Each position is assigned to exactly one node (The "AllDifferent" fix)
    for p in positions:
        model.addConstr(gp.quicksum(x[u, p] for u in nodes) == 1, name=f"pos_{p}_assign")
        
    # Link pos[u] to the binary assignment matrix
    for u in nodes:
        model.addConstr(pos[u] == gp.quicksum(p * x[u, p] for p in positions), name=f"link_{u}")
    
    # Auxiliary variables for absolute differences
    edge_list = list(graph.edges())
    num_edges = len(edge_list)
    abs_diff = model.addVars(num_edges, lb=0, vtype=GRB.CONTINUOUS, name="abs_diff")
    
    # For each edge, add constraints for absolute difference
    for i, (u, v) in enumerate(edge_list):
        # Create auxiliary variables for the difference
        diff_var = model.addVar(lb=-(n-1), ub=n-1, vtype=GRB.CONTINUOUS, name=f"diff_{i}")
        
        # diff_var = pos[u] - pos[v]
        model.addConstr(diff_var == pos[u] - pos[v])
        
        # abs_diff[i] = |diff_var|
        model.addConstr(abs_diff[i] >= diff_var)
        model.addConstr(abs_diff[i] >= -diff_var)
    
    # Objective: minimize sum of absolute differences
    objective = gp.quicksum(abs_diff[i] for i in range(num_edges))
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        ordering = [None] * n
        for u in nodes:
            # Find the position p where x[u, p] is 1
            for p in positions:
                if x[u, p].X > 0.5:  # Check binary threshold
                    ordering[p] = u
                    break
        
        objective_value = int(round(model.objVal))
        status = "OPTIMAL" if model.status == GRB.OPTIMAL else "SUBOPTIMAL"
        
        return status, ordering, objective_value
    else:
        return "NOT_SOLVED", [], float('inf')
