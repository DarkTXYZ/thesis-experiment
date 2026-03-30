import networkx as nx

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
    
    return int(max(edges, degree)) + 1