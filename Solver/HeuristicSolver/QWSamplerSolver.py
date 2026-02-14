import sys
import os
import warnings
from collections import defaultdict
from typing import Tuple, Dict, Any

import numpy as np
import networkx as nx
import dimod
from dwave.samplers import PathIntegralAnnealingSampler
from pyqubo import Array

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_class import BaseSolver, SolverResult
from penalty_coefficients import calculate_lucas_bound, calculate_exact_bound

class QWaveSamplerSolver(BaseSolver):
    """Quantum Wave Sampler Solver using Path Integral Annealing."""
    
    def __init__(self):
        # Sampling parameters
        self.num_reads: int = 10
        self.num_sweeps: int = 1000
        self.seed: int | None = None
        
        # Penalty parameters
        self.penalty_mode: str = "lucas"
        self.penalty_thermometer: float = 1.0
        self.penalty_bijective: float = 1.0
        
        # Beta schedule parameters
        self.beta_schedule_type: str = 'default'
        self.beta_range: Tuple[float, float] = (0.0, 1.0)
        self.use_auto_beta_range: bool = False
        self.actual_beta_range: Tuple[float, float] | None = None
    
    def configure(self, **kwargs) -> None:
        """Configure solver parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # =========================================================================
    # QUBO FORMULATION
    # =========================================================================
    
    def _build_qubo(self, graph: nx.Graph) -> dimod.BinaryQuadraticModel:
        """
        Build QUBO formulation for MINLA problem.
        
        Uses thermometer encoding with binary variables X[u][k]:
        - X[u][k] = 1 if node u is placed at position <= k
        - Thermometer constraint: X[u][k] >= X[u][k+1]
        - Bijective constraint: Each position has exactly one node
        - Objective: Minimize sum of edge distances
        
        Args:
            graph: Input graph
            
        Returns:
            Binary quadratic model representing the QUBO
        """
        n = graph.number_of_nodes()
        mu_thermo, mu_bijec = self._get_penalties(graph)
        
        X = Array.create('X', shape=(n, n), vartype='BINARY')
        
        # Thermometer encoding constraint
        H_thermometer = mu_thermo * sum(
            (1 - X[u][k]) * X[u][k+1]
            for u in range(n)
            for k in range(n-1)
        )
        
        # Bijective constraint (each position assigned exactly once)
        H_bijective = mu_bijec * sum(
            ((n - k) - sum(X[u][k] for u in range(n))) ** 2
            for k in range(n)
        )
        
        # MINLA objective (minimize sum of edge distances)
        H_objective = sum(
            sum(X[u][k] + X[v][k] - 2 * X[u][k] * X[v][k] for k in range(n))
            for u, v in graph.edges
        )
        
        H = H_thermometer + H_bijective + H_objective
        model = H.compile()
        return model.to_bqm()
    
    def _get_penalties(self, graph: nx.Graph) -> Tuple[float, float]:
        """Get penalty coefficients based on configured penalty mode."""
        if self.penalty_mode == "manual":
            return self.penalty_thermometer, self.penalty_bijective
        elif self.penalty_mode == "lucas":
            return calculate_lucas_bound(graph)
        elif self.penalty_mode == "exact":
            return calculate_exact_bound(graph)
        raise ValueError(f"Unknown penalty mode: {self.penalty_mode}")
    
    def _extract_Q_matrix(self, bqm) -> tuple[dict, float]:
        """Extract Q matrix and offset from BQM."""
        Q = {}
        for var, bias in bqm.linear.items():
            Q[(var, var)] = bias
        for (i, j), bias in bqm.quadratic.items():
            Q[(i, j)] = bias
        return Q, bqm.offset
    
    # =========================================================================
    # BETA RANGE CALCULATION
    # =========================================================================
    
    def default_ising_beta_range(
        self, 
        h: Dict, 
        J: Dict, 
        max_single_qubit_excitation_rate: float = 0.01,
        scale_T_with_N: bool = True
    ) -> list[float]:
        """Calculate default beta range for Ising model."""
        if not 0 < max_single_qubit_excitation_rate < 1:
            raise ValueError('Targeted single qubit excitations rates must be in range (0,1)')

        sum_abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
        if sum_abs_bias_dict:
            min_abs_bias_dict = {k: v for k, v in sum_abs_bias_dict.items() if v != 0}
        else:
            min_abs_bias_dict = {}
            
        for (k1, k2), v in J.items():
            for k in [k1, k2]:
                sum_abs_bias_dict[k] += abs(v)
                if v != 0:
                    if k in min_abs_bias_dict:
                        min_abs_bias_dict[k] = min(abs(v), min_abs_bias_dict[k])
                    else:
                        min_abs_bias_dict[k] = abs(v)

        if not min_abs_bias_dict:
            warn_msg = ('All bqm biases are zero (all energies are zero), this is '
                        'likely a value error. Temperature range is set arbitrarily '
                        'to [0.1,1]. Metropolis-Hastings update is non-ergodic.')
            warnings.warn(warn_msg)
            return [0.1, 1]

        max_effective_field = max(sum_abs_bias_dict.values(), default=0)

        if max_effective_field == 0:
            hot_beta = 1
        else:
            hot_beta = np.log(2) / (2 * max_effective_field)

        if len(min_abs_bias_dict) == 0:
            cold_beta = hot_beta
        else:
            values_array = np.array(list(min_abs_bias_dict.values()), dtype=float)
            min_effective_field = np.min(values_array)
            if scale_T_with_N:
                number_min_gaps = np.sum(min_effective_field == values_array)
            else:
                number_min_gaps = 1
            cold_beta = np.log(number_min_gaps / max_single_qubit_excitation_rate) / (2 * min_effective_field)

        return [hot_beta, cold_beta]
    
    def default_bqm_beta_range(self, bqm: dimod.BinaryQuadraticModel) -> list[float]:
        """Calculate default beta range from BQM by converting to Ising."""
        ising = bqm.spin
        return self.default_ising_beta_range(ising.linear, ising.quadratic)
    
    # =========================================================================
    # SCHEDULE GENERATION
    # =========================================================================

    def generate_linear_schedule(
        self, 
        steps: int, 
        beta_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear annealing schedule.
        
        Hp_field: increases linearly from hot_beta to cold_beta
        Hd_field: decreases linearly from cold_beta to 0
        """
        hot_beta, cold_beta = beta_range
        s = np.linspace(0, 1, steps)  # annealing parameter s ∈ [0, 1]
        
        Hp_field = hot_beta + (cold_beta - hot_beta) * s  # hot_beta → cold_beta
        Hd_field = cold_beta * (1 - s)                     # cold_beta → 0
        
        return Hp_field, Hd_field
    
    def generate_exponential_schedule(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate exponential annealing schedule."""
        s = np.linspace(1, steps, steps)
        Hd_field = np.exp(-0.01 * s)
        Hp_field = 1 - Hd_field
        return Hp_field, Hd_field
    
    def generate_power_schedule(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate power-law annealing schedule."""
        s = np.linspace(1, steps, steps)
        Hd_field = np.power(s, -0.3)
        Hp_field = 1 - Hd_field
        return Hp_field, Hd_field
    
    # =========================================================================
    # SAMPLING AND SOLUTION
    # =========================================================================
    
    def _get_sampler(self) -> PathIntegralAnnealingSampler:
        """Get the path integral annealing sampler."""
        return PathIntegralAnnealingSampler()

    def solve(self, graph: nx.Graph) -> SolverResult:
        """Solve the MINLA problem using path integral annealing."""
        bqm = self._build_qubo(graph)
        sampler = self._get_sampler()
        
        # Determine beta range
        if self.use_auto_beta_range:
            self.actual_beta_range = tuple(self.default_bqm_beta_range(bqm))
        else:
            self.actual_beta_range = self.beta_range
        
        # Prepare sampling parameters
        sample_kwargs = {
            'num_reads': self.num_reads, 
            'num_sweeps': self.num_sweeps,
        }
        if self.seed is not None:
            sample_kwargs['seed'] = self.seed
        
        # Sample based on schedule type
        response = self._sample_with_schedule(sampler, bqm, sample_kwargs)
        
        # Extract and decode best solution
        best_sample = response.first.sample
        energy = response.first.energy
        ordering, is_feasible = self._decode_solution(best_sample, graph.number_of_nodes())
        
        return SolverResult(energy, ordering, is_feasible)
    
    def _sample_with_schedule(
        self, 
        sampler: PathIntegralAnnealingSampler,
        bqm: dimod.BinaryQuadraticModel,
        sample_kwargs: Dict[str, Any]
    ):
        """Sample BQM with the configured beta schedule type."""
        schedule_type = self.beta_schedule_type
        
        # Custom schedules with explicit Hp and Hd fields
        if schedule_type == 'linear_beta':
            Hp, Hd = self.generate_linear_schedule(self.num_sweeps, self.actual_beta_range)
            return sampler.sample(bqm, beta_schedule_type='custom',
                                Hp_field=Hp, Hd_field=Hd, **sample_kwargs)
        
        elif schedule_type == 'exponential':
            Hp, Hd = self.generate_exponential_schedule(self.num_sweeps)
            return sampler.sample(bqm, beta_schedule_type='custom',
                                Hp_field=Hp, Hd_field=Hd, **sample_kwargs)
        
        elif schedule_type == 'power':
            Hp, Hd = self.generate_power_schedule(self.num_sweeps)
            return sampler.sample(bqm, beta_schedule_type='custom',
                                Hp_field=Hp, Hd_field=Hd, **sample_kwargs)
        
        # Built-in schedule types
        elif schedule_type in ['geometric', 'linear']:
            return sampler.sample(bqm, beta_schedule_type=schedule_type, **sample_kwargs)
        
        # Default schedule
        else:
            return sampler.sample(bqm, **sample_kwargs)

    def _decode_solution(self, raw_sample: Dict, num_nodes: int) -> Tuple[np.ndarray, bool]:
        """Decode raw sample into ordering and check feasibility."""
        n = num_nodes
        sol = np.array([
            [raw_sample.get(f'X[{u}][{k}]', 0) for k in range(n)]
            for u in range(n)
        ], dtype=int)
        
        is_feasible = self._check_feasibility(sol, n)
        ordering = np.sum(sol, axis=1)
        
        return ordering, is_feasible
    
    def _check_feasibility(self, sol: np.ndarray, n: int) -> bool:
        """Check if solution satisfies MINLA constraints."""
        # Check thermometer encoding constraint
        for u in range(n):
            if np.any((sol[u, :-1] == 0) & (sol[u, 1:] == 1)):
                return False
        
        # Check bijective constraint (all positions must be unique)
        labels = set(np.sum(sol, axis=1))
        if labels != set(range(1, n + 1)):
            return False
        
        return True


if __name__ == "__main__":
    solver = QWaveSamplerSolver()
    print(solver.generate_power_schedule(1000))
