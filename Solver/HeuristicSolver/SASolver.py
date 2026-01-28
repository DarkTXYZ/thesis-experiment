import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_class import BaseSolver, SolverResult
from dwave.samplers import PathIntegralAnnealingSampler, SteepestDescentSampler
from pyqubo import Array
import networkx as nx
import numpy as np
from penalty_coefficients import calculate_lucas_bound, calculate_exact_bound

class QWaveSamplerSolver(BaseSolver):
    
    def __init__(self):
        self.num_reads = 100
        self.num_sweeps = 1000
        self.sampler_type = "path"
        self.penalty_mode = "lucas"
        self.penalty_thermometer = 1.0
        self.penalty_bijective = 1.0
        self.seed = None
    
    def configure(self, **kwargs) -> None:
        self.num_reads = kwargs.get('num_reads', self.num_reads)
        self.num_sweeps = kwargs.get('num_sweeps', self.num_sweeps)
        self.sampler_type = kwargs.get('sampler_type', self.sampler_type)
        self.penalty_mode = kwargs.get('penalty_mode', self.penalty_mode)
        self.penalty_thermometer = kwargs.get('penalty_thermometer', self.penalty_thermometer)
        self.penalty_bijective = kwargs.get('penalty_bijective', self.penalty_bijective)
        self.seed = kwargs.get('seed', self.seed)
    
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
    
    def _extract_Q_matrix(self, bqm) -> tuple[dict, float]:
        Q = {}
        
        for var, bias in bqm.linear.items():
            Q[(var, var)] = bias
        
        for (i, j), bias in bqm.quadratic.items():
            Q[(i, j)] = bias
        
        return Q, bqm.offset
    
    def _get_sampler(self):
        if self.sampler_type == "steepest":
            return SteepestDescentSampler()
        elif self.sampler_type == "path":
            return PathIntegralAnnealingSampler()
        raise ValueError(f"Unknown sampler type: {self.sampler_type}")

    def solve(self, graph: nx.Graph) -> SolverResult:
        bqm = self._build_qubo(graph)
        
        sampler = self._get_sampler()
        
        sample_kwargs = {'num_reads': self.num_reads, 'num_sweeps': self.num_sweeps}
        if self.seed is not None:
            sample_kwargs['seed'] = self.seed
        response = sampler.sample(bqm, **sample_kwargs)
        
        best_sample = response.first.sample
        energy = response.first.energy
        ordering, is_feasible = self._decode_solution(best_sample, graph.number_of_nodes())
        
        return SolverResult(energy, ordering, is_feasible)

    def _decode_solution(self, raw_sample: dict, num_nodes: int) -> tuple[np.ndarray, bool]:
        n = num_nodes
        sol = np.array([
            [raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
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
    graph = nx.erdos_renyi_graph(5, 0.5, seed=42)
    
    solver = QWaveSamplerSolver()
    solver.configure(num_reads=100, num_sweeps=1000, sampler_type="path")
    
    result = solver.solve(graph)
    print("Energy:", result.energy)
    print("Ordering:", result.ordering)
    print("Feasible:", result.is_feasible)  