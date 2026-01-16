import networkx as nx

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
        # print(f"Edges Remaining: {number_of_edges_remaining}, Possible at length {edge_length}: {possible_edges_at_length}")
        if (number_of_edges_remaining >= possible_edges_at_length):
            edges += possible_edges_at_length * edge_length
            number_of_edges_remaining -= possible_edges_at_length
            edge_length -= 1
        else:
            edges += number_of_edges_remaining * edge_length
            number_of_edges_remaining = 0
            edge_length -= 1
    
    # print(f"\tUpper Bound Calculation: naive={naive}, complete={complete}, edges={edges}")
    
    return int(min(naive, complete,edges)) + 1

def calculate_lower_obj_bound(G: nx.Graph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    number_of_edges_remaining = m
    edges = 0
    edge_length = 1
    while number_of_edges_remaining > 0:
        possible_edges_at_length = n - edge_length
        # print(f"Edges Remaining: {number_of_edges_remaining}, Possible at length {edge_length}: {possible_edges_at_length}")
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
    # lower_bound_obj = calculate_lower_obj_bound(G)
    
    # if upper_bound_obj == lower_bound_obj:
    #     mu_thermometer = upper_bound_obj + 1
    #     mu_bijective = upper_bound_obj + 1
    # else:
    #     mu_thermometer = upper_bound_obj - lower_bound_obj + 1
    #     mu_bijective = upper_bound_obj - lower_bound_obj + 1
    
    return upper_bound_obj + 1, upper_bound_obj + 1

def calculate_lucas_bound(G: nx.Graph):
    degree_sequence = (d for n, d in G.degree())
    delta = max(degree_sequence)
    
    mu_thermometer = delta + 1
    mu_bijective = delta + 1
    
    # print(f"\tLucas Bound Calculation: delta={delta}")
    
    return mu_thermometer, mu_bijective
    

if __name__ == "__main__":
    G = nx.Graph()
    # G.add_edges_from([(0, 1), (1, 2), (0, 2)])

    # mu_thermo, mu_bijec = calculate_exact_bound(G)
    # print(f"Exact Bound - Mu Thermometer: {mu_thermo}, Mu Bijective: {mu_bijec}")
    # mu_thermo_lucas, mu_bijec_lucas = calculate_lucas_bound(G)
    # print(f"Lucas Bound - Mu Thermometer: {mu_thermo_lucas}, Mu Bijective: {mu_bijec_lucas}")