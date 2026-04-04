import networkx as nx
import numpy as np
import math

try:
    import cvxpy as cp
except ImportError:
    cp = None

class MaxLAUpperBounds:
    def __init__(self, G: nx.Graph):
        self.G = nx.Graph(G)
        self.n = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()

    def naive_method(self):
        return self.m * (self.n - 1)

    def edges_method_closed_form(self):
        n, m = self.n, self.m
        F = lambda d: (n - d) * (n - d + 1) / 2
        d_star = math.ceil(n + 0.5 - 0.5 * math.sqrt(8 * m + 1))
        return float(
            (m - F(d_star)) * (d_star - 1)
            + (1 / 6) * (n - d_star) * (n * n + (n + 3) * d_star - 2 * d_star * d_star - 1)
        )

    def degree_sequence_bound(self):
        n = self.n
        degrees = sorted((deg for _, deg in self.G.degree()), reverse=True)
        b_vals = sorted((max(i - 1, n - i) for i in range(1, n + 1)), reverse=True)
        return float(0.5 * sum(d * b for d, b in zip(degrees, b_vals)))
    
    def maxcut_maximum_degree_upper_bound(self):
        max_degree = max(dict(self.G.degree()).values())
        return float(max_degree * self.n / 2)
    
    def maxla_upper_bound_via_maximum_degree_maxcut(self):
        return (self.n - 1) * self.maxcut_maximum_degree_upper_bound()

    def maxcut_spectral_upper_bound(self):
        L = nx.laplacian_matrix(self.G).astype(float).toarray()
        lambda_max = np.linalg.eigvalsh(L).max()
        return float(self.n * lambda_max / 4.0)
    
    def maxla_upper_bound_via_spectral_maxcut(self):
        return (self.n - 1) * self.maxcut_spectral_upper_bound()

    def maxcut_sdp_upper_bound(self, solver=None):
        if cp is None:
            raise ImportError(
                "cvxpy is required for the SDP bound. Install with: pip install cvxpy"
            )

        n = self.n
        if n == 0:
            return 0.0

        L = nx.laplacian_matrix(self.G).astype(float).toarray()
        X = cp.Variable((n, n), symmetric=True)
        constraints = [cp.diag(X) == 1, X >> 0]
        objective = cp.Maximize(0.25 * cp.trace(L @ X))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, verbose=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SDP did not solve successfully. Status: {problem.status}")

        return float(problem.value)

    def maxla_upper_bound_via_sdp_maxcut(self, solver=None):
        return (self.n - 1) * self.maxcut_sdp_upper_bound(solver=solver)

    def evaluate_all(self, include_exact_cut_methods=False, include_sdp=False, solver=None):
        results = {
            "Naive Method": self.naive_method(),
            "Edges Method Closed Form": self.edges_method_closed_form(),
            "Degree-Sequence Bound": self.degree_sequence_bound(),
            "Maximum Degree Bound": self.maxla_upper_bound_via_maximum_degree_maxcut(),
            "Spectral MaxCut Bound": self.maxla_upper_bound_via_spectral_maxcut(),
        }

        if include_exact_cut_methods:
            results["Exact MaxCut Bound"] = self.maxla_upper_bound_via_exact_maxcut()
            results["Exact Cut-Profile Bound"] = self.cut_profile_bound_exact()

        if include_sdp:
            results["SDP MaxCut Bound"] = self.maxla_upper_bound_via_sdp_maxcut(solver=solver)

        return results

    def bound(self, include_exact_cut_methods=False, include_sdp=False, solver=None):
        return min(self.evaluate_all(
            include_exact_cut_methods=include_exact_cut_methods,
            include_sdp=include_sdp,
            solver=solver
        ).values())


if __name__ == "__main__":
    G = nx.erdos_renyi_graph(25, 0.5, seed = 42)
    j = MaxLAUpperBounds(G).bound(include_sdp=True, solver=cp.SCS)
    print("Upper Bound:", j)
