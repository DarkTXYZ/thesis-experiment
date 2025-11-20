import networkx as nx
from typing import List

def calculate_min_linear_arrangement(graph: nx.Graph, ordering: List[int]):
    """Calculate the minimum linear arrangement cost for a given ordering."""
    position = {vertex: label for vertex, label in enumerate(ordering)}
    cost = 0
    for u, v in graph.edges():
        cost += abs(position[u] - position[v])
    return cost