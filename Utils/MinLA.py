import dimod
import networkx as nx
from pyqubo import Array
from typing import List

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