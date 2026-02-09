import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from penalty_coefficients import calculate_exact_bound, calculate_lucas_bound
from solver_class import BaseSolver, SolverResult
import networkx as nx
import numpy as np
import openjij as oj


class OpenJijSolver(BaseSolver):
    
    def __init__(self):
        self.num_reads = 100
        self.use_sparse = True
        self.penalty_mode = "lucas"
        self.penalty_thermometer = 1.0
        self.penalty_bijective = 1.0
        self.seed = None
    
    def configure(self, **kwargs) -> None:
        if 'num_reads' in kwargs:
            self.num_reads = kwargs['num_reads']
        if 'use_sparse' in kwargs:
            self.use_sparse = kwargs['use_sparse']
        if 'penalty_mode' in kwargs:
            self.penalty_mode = kwargs['penalty_mode']
        if 'penalty_thermometer' in kwargs:
            self.penalty_thermometer = kwargs['penalty_thermometer']
        if 'penalty_bijective' in kwargs:
            self.penalty_bijective = kwargs['penalty_bijective']
        if 'seed' in kwargs:
            self.seed = kwargs['seed']
    
    def _build_qubo(self, graph: nx.Graph):
        from pyqubo import Array

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
        elif self.penalty_mode == "exact":
            return calculate_exact_bound(graph)
        elif self.penalty_mode == "lucas":
            return calculate_lucas_bound(graph)
        raise ValueError(f"Unknown penalty mode: {self.penalty_mode}")

    def _extract_Q_matrix(self, bqm) -> tuple[dict, float]:
        Q = {}
        
        for var, bias in bqm.linear.items():
            Q[(var, var)] = bias
        
        for (i, j), bias in bqm.quadratic.items():
            Q[(i, j)] = bias
        
        return Q, bqm.offset

    def solve(self, graph: nx.Graph) -> SolverResult:
        bqm = self._build_qubo(graph)
        Q, offset = self._extract_Q_matrix(bqm)

        sampler = oj.SQASampler()
        sample_kwargs = {'num_reads': self.num_reads, 'sparse': self.use_sparse}
        if self.seed is not None:
            sample_kwargs['seed'] = self.seed
        response = sampler.sample_qubo(Q, **sample_kwargs)
        
        best_sample = response.first.sample
        energy = response.first.energy + offset
        ordering, is_feasible = self._decode_solution(best_sample, graph.number_of_nodes())
        
        return SolverResult(energy, ordering, is_feasible, raw_sample=best_sample)

    def _decode_solution(self, raw_sample: dict, num_nodes: int) -> tuple[np.ndarray, bool]:
        n = num_nodes
        sol = np.array([
            [raw_sample[f'X[{u}][{k}]'] for k in range(n)]
            for u in range(n)
        ], dtype=int)
        
        is_feasible = self._check_feasibility(sol, n)
        ordering = np.sum(sol, axis=1)
        
        return ordering, is_feasible
    
    def _check_feasibility(self, sol: np.ndarray, n: int) -> bool:
        for u in range(n):
            if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
                return False
        
        labels = set(np.sum(sol, axis=1))
        if labels != set(range(1, n + 1)):
            return False
        
        return True
    
if __name__ == "__main__":
    n = 5
    graph = nx.erdos_renyi_graph(n, 0.5, seed=42)
    
    solver = OpenJijSolver()
    solver.configure(num_reads=10, penalty_mode="lucas", use_sparse=True)
    result = solver.solve(graph)
    print("Energy:", result.energy)
    print("Ordering:", result.ordering)
    print("Feasible:", result.is_feasible)