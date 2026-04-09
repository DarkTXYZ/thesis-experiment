import networkx as nx
import numpy as np
import math
from .max_la_upper_bound import MaxLAUpperBounds

class MinLALowerBounds:
    def __init__(self, G):
        self.G = G
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()

    def edges_method(self):
        n, m = self.n, self.m
        i, f, lb = 1, 0, 0
        
        while f + n - i <= m:
            if n == i:
                break
            f += n - i
            lb += i * (n - i)
            i += 1

        return lb + i * (m - f)

    def degree_method(self):
        lb = sum(
            (d * d / 4 + d / 2) if d % 2 == 0 else ((d * d + 2 * d + 1) / 4)
            for d in (self.G.degree(v) for v in self.G.nodes())
        )
        return lb / 2

    def juvan_mohar_method(self):
        if self.n <= 1:
            return 0
        
        L = nx.laplacian_matrix(self.G).astype(float)
        e = np.linalg.eigvals(L.toarray())
        lambda_2 = sorted(e)[1]

        return lambda_2 * (self.n ** 2 - 1) / 6

    def gomory_hu_method(self):
        G_copy = self.G.copy()
        nx.set_edge_attributes(G_copy, 1, "capacity")
        T = nx.gomory_hu_tree(G_copy)
        return sum(data['weight'] for _, _, data in T.edges(data=True))

    def path_method(self):
        k = math.floor(self.n - ((2*self.n - 1) ** 2 - 8 * self.m) ** 0.5 / 2 - 1 / 2)
        return k * (k + 1) * (3*self.n - 2*k - 1) / 6
    
    def complement_graph_max_la(self):
        ub = MaxLAUpperBounds(nx.complement(self.G)).bound()
        return self.n * (self.n*self.n - 1) / 6 - ub

    def evaluate_all(self):
        results = {
            "Edges Method": self.edges_method(),
            "Complement Graph Method": self.complement_graph_max_la(),
            "Degree Method": self.degree_method(),
            "Juvan-Mohar Method": self.juvan_mohar_method(),
            "Gomory-Hu Method": self.gomory_hu_method(),
            "Path Method": self.path_method(),
        }
        return results
    
    def bound(self):
        return max(self.evaluate_all().values())

def calculate_lower_obj_bound(G: nx.Graph):
    return MinLALowerBounds(G).bound()

if __name__ == "__main__":
    G = nx.erdos_renyi_graph(25, 0.5, seed = 42)
    j = calculate_lower_obj_bound(G)
    print("Lower Bound:", j)
