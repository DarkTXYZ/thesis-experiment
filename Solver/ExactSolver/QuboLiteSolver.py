import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_class import BaseSolver, SolverResult
from qubolite import qubo, solving
from pyqubo import Array
import networkx as nx
import numpy as np
from penalty_coefficients import calculate_lucas_bound, calculate_exact_bound


class QuboLiteSolver(BaseSolver):
    
    def __init__(self):
        self.max_threads = 1
        self.penalty_mode = "lucas"
        self.penalty_thermometer = 1.0
        self.penalty_bijective = 1.0
    
    def configure(self, **kwargs) -> None:
        self.max_threads = kwargs.get('max_threads', self.max_threads)
        self.penalty_mode = kwargs.get('penalty_mode', self.penalty_mode)
        self.penalty_thermometer = kwargs.get('penalty_thermometer', self.penalty_thermometer)
        self.penalty_bijective = kwargs.get('penalty_bijective', self.penalty_bijective)
    
    def _build_qubo(self, graph: nx.Graph):
        n = graph.number_of_nodes()
        mu_thermo, mu_bijec = self._get_penalties(graph)
        
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
    
    def _get_penalties(self, graph: nx.Graph) -> tuple[float, float]:
        if self.penalty_mode == "manual":
            return self.penalty_thermometer, self.penalty_bijective
        elif self.penalty_mode == "lucas":
            return calculate_lucas_bound(graph)
        elif self.penalty_mode == "exact":
            return calculate_exact_bound(graph)
        raise ValueError(f"Unknown penalty mode: {self.penalty_mode}")
    
    def _extract_Q_matrix(self, bqm, n: int) -> tuple[np.ndarray, float]:
        Q = np.zeros((n * n, n * n))
        
        for var, bias in bqm.linear.items():
            idx = self._var_to_index(var, n)
            Q[idx, idx] = bias
        
        for (vi, vj), bias in bqm.quadratic.items():
            idx_i = self._var_to_index(vi, n)
            idx_j = self._var_to_index(vj, n)
            Q[idx_i, idx_j] = bias
        
        return Q, bqm.offset
    
    def _var_to_index(self, var_name: str, n: int) -> int:
        parts = var_name.replace('X[', '').replace(']', '').split('[')
        u, k = int(parts[0]), int(parts[1])
        return u * n + k

    def solve(self, graph: nx.Graph) -> SolverResult:
        n = graph.number_of_nodes()
        bqm = self._build_qubo(graph)
        Q_matrix, offset = self._extract_Q_matrix(bqm, n)
        
        Q = qubo(Q_matrix)
        solution, energy = solving.brute_force(Q, max_threads=self.max_threads)
        
        ordering, is_feasible = self._decode_solution(solution, n)
        
        return SolverResult(energy + offset, ordering, is_feasible)

    def _decode_solution(self, solution: np.ndarray, num_nodes: int) -> tuple[np.ndarray, bool]:
        n = num_nodes
        sol = solution.reshape((n, n))
        
        is_feasible = self._check_feasibility(sol, n)
        ordering = np.sum(sol, axis=1).astype(int)
        
        return ordering, is_feasible
    
    def _check_feasibility(self, sol: np.ndarray, n: int) -> bool:
        for u in range(n):
            if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
                return False
        
        labels = set(np.sum(sol, axis=1).astype(int))
        if labels != set(range(1, n + 1)):
            return False
        
        return True


if __name__ == "__main__":
    graph = nx.erdos_renyi_graph(4, 0.5, seed=42)
    
    solver = QuboLiteSolver()
    solver.configure(max_threads=1)
    
    result = solver.solve(graph)
    print("Energy:", result.energy)
    print("Ordering:", result.ordering)
    print("Feasible:", result.is_feasible)
