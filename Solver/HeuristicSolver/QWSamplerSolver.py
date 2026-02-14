import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_class import BaseSolver, SolverResult
from dwave.samplers import PathIntegralAnnealingSampler
from pyqubo import Array
import networkx as nx
import dimod
import numpy as np
from penalty_coefficients import calculate_lucas_bound, calculate_exact_bound
import numpy as np
from collections import defaultdict

import warnings

class QWaveSamplerSolver(BaseSolver):
    
    def __init__(self):
        self.num_reads = 10
        self.num_sweeps = 1000
        self.sampler_type = "path"
        self.penalty_mode = "lucas"
        self.penalty_thermometer = 1.0
        self.penalty_bijective = 1.0
        self.seed = None
        self.beta_schedule_type = 'default'
        self.beta_range = (0.0, 1.0)
        self.use_auto_beta_range = False
    
    def configure(self, **kwargs) -> None:
        if 'num_reads' in kwargs:
            self.num_reads = kwargs['num_reads']
        if 'num_sweeps' in kwargs:
            self.num_sweeps = kwargs['num_sweeps']
        if 'sampler_type' in kwargs:
            self.sampler_type = kwargs['sampler_type']
        if 'penalty_mode' in kwargs:
            self.penalty_mode = kwargs['penalty_mode']
        if 'penalty_thermometer' in kwargs:
            self.penalty_thermometer = kwargs['penalty_thermometer']
        if 'penalty_bijective' in kwargs:
            self.penalty_bijective = kwargs['penalty_bijective']
        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        if 'beta_schedule_type' in kwargs:
            self.beta_schedule_type = kwargs['beta_schedule_type']
        if 'beta_range' in kwargs:
            self.beta_range = kwargs['beta_range']
        if 'use_auto_beta_range' in kwargs:
            self.use_auto_beta_range = kwargs['use_auto_beta_range']

    def _build_qubo(self, graph: nx.Graph) -> dimod.BinaryQuadraticModel:
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
        return PathIntegralAnnealingSampler()
    
    def default_ising_beta_range(self, h, J, max_single_qubit_excitation_rate = 0.01,
                              scale_T_with_N = True):
        if not 0 < max_single_qubit_excitation_rate < 1:
            raise ValueError('Targeted single qubit excitations rates must be in range (0,1)')

        sum_abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
        if sum_abs_bias_dict:
            min_abs_bias_dict = {k: v for k, v in sum_abs_bias_dict.items() if v != 0}
        else:
            min_abs_bias_dict = {}
        for (k1, k2), v in J.items():
            for k in [k1,k2]:
                sum_abs_bias_dict[k] += abs(v)
                if v != 0:
                    if k in min_abs_bias_dict:
                        min_abs_bias_dict[k] = min(abs(v),min_abs_bias_dict[k])
                    else:
                        min_abs_bias_dict[k] = abs(v)

        if not min_abs_bias_dict:
            warn_msg = ('All bqm biases are zero (all energies are zero), this is '
                        'likely a value error. Temperature range is set arbitrarily '
                        'to [0.1,1]. Metropolis-Hastings update is non-ergodic.')
            warnings.warn(warn_msg)
            return([0.1,1])


        max_effective_field = max(sum_abs_bias_dict.values(), default=0)

        if max_effective_field == 0:
            hot_beta = 1
        else:
            hot_beta = np.log(2) / (2*max_effective_field)

        if len(min_abs_bias_dict)==0:
            cold_beta = hot_beta
        else:
            values_array = np.array(list(min_abs_bias_dict.values()),dtype=float)
            min_effective_field = np.min(values_array)
            if scale_T_with_N:
                number_min_gaps = np.sum(min_effective_field == values_array)
            else:
                number_min_gaps = 1
            cold_beta = np.log(number_min_gaps/max_single_qubit_excitation_rate) / (2*min_effective_field)

        return [hot_beta, cold_beta]
    
    def default_bqm_beta_range(self, bqm):
        ising = bqm.spin
        return self.default_ising_beta_range(ising.linear, ising.quadratic)

    def _calculate_beta_range(self, bqm) -> tuple[float, float]:
        """
        Calculate appropriate hot_beta and cold_beta from the BQM.
        
        hot_beta: Low β (high T) - allows exploration, 50% flip probability
        cold_beta: High β (low T) - suppresses excitations, 1% excitation rate
        
        Returns:
            (hot_beta, cold_beta)
        """
        from collections import defaultdict
        
        # Calculate sum of absolute biases and min absolute bias for each variable
        sum_abs_bias = defaultdict(float)
        min_abs_bias = defaultdict(lambda: float('inf'))
        
        # Add linear biases
        for var, bias in bqm.linear.items():
            abs_bias = abs(bias)
            sum_abs_bias[var] += abs_bias
            if abs_bias > 0:
                min_abs_bias[var] = min(min_abs_bias[var], abs_bias)
        
        # Add quadratic biases
        for (var1, var2), bias in bqm.quadratic.items():
            abs_bias = abs(bias)
            sum_abs_bias[var1] += abs_bias
            sum_abs_bias[var2] += abs_bias
            if abs_bias > 0:
                min_abs_bias[var1] = min(min_abs_bias[var1], abs_bias)
                min_abs_bias[var2] = min(min_abs_bias[var2], abs_bias)
        
        # Hot beta: Allow 50% flip probability for worst case
        max_effective_field = max(sum_abs_bias.values(), default=0)
        if max_effective_field == 0:
            hot_beta = 1.0
        else:
            hot_beta = np.log(2) / (2 * max_effective_field)
        
        # Cold beta: Suppress excitations to 1%
        if len(min_abs_bias) == 0:
            cold_beta = hot_beta
        else:
            values_array = np.array([v for v in min_abs_bias.values() if v < float('inf')])
            if len(values_array) == 0:
                cold_beta = hot_beta
            else:
                min_effective_field = np.min(values_array)
                number_min_gaps = np.sum(min_effective_field == values_array)
                max_excitation_rate = 0.01
                cold_beta = np.log(number_min_gaps / max_excitation_rate) / (2 * min_effective_field)
        
        return (hot_beta, cold_beta)

    def generate_linear_schedule(self, steps: int, beta_range: tuple[float, float]):
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
    
    def generate_exponential_schedule(self, steps):
        s = np.linspace(1, steps, steps)

        Hd_field = np.exp(-0.01 * s)
        Hp_field = 1 - Hd_field

        return Hp_field, Hd_field
    
    def generate_power_schedule(self, steps):
        s = np.linspace(1, steps, steps)

        Hd_field = np.power(s, -0.3)
        Hp_field = 1 - Hd_field

        return Hp_field, Hd_field

    def solve(self, graph: nx.Graph) -> SolverResult:
        bqm = self._build_qubo(graph)
        
        sampler = self._get_sampler()
        
        # Determine beta range - auto-calculate or use configured values
        if self.use_auto_beta_range:
            # Calculate from BQM and store in solver instance
            self.actual_beta_range = self.default_bqm_beta_range(bqm)
        else:
            # Use configured beta_range from config
            self.actual_beta_range = self.beta_range
        
        sample_kwargs = {
            'num_reads': self.num_reads, 
            'num_sweeps': self.num_sweeps,
        }

        if self.seed is not None:
            sample_kwargs['seed'] = self.seed

        if self.beta_schedule_type == 'linear_beta':
            Hp, Hd = self.generate_linear_schedule(self.num_sweeps, self.actual_beta_range)
            response = sampler.sample(bqm, 
                            beta_schedule_type='custom',
                            Hp_field=Hp,
                            Hd_field=Hd,
                        **sample_kwargs)
        elif self.beta_schedule_type == 'exponential':
            Hp, Hd = self.generate_exponential_schedule(self.num_sweeps)
            response = sampler.sample(bqm, 
                            beta_schedule_type='custom',
                            Hp_field=Hp,
                            Hd_field=Hd,
                        **sample_kwargs)
        elif self.beta_schedule_type == 'power':
            Hp, Hd = self.generate_power_schedule(self.num_sweeps)
            response = sampler.sample(bqm, 
                            beta_schedule_type='custom',
                            Hp_field=Hp,
                            Hd_field=Hd,
                        **sample_kwargs)
        elif self.beta_schedule_type == 'geometric':
            response = sampler.sample(bqm, 
                            beta_schedule_type='geometric',
                            **sample_kwargs)
        elif self.beta_schedule_type == 'linear':
            response = sampler.sample(bqm, 
                            beta_schedule_type='linear',
                            **sample_kwargs)
        elif self.beta_schedule_type == 'default':
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
    solver = QWaveSamplerSolver()
    print(solver.generate_power_schedule(1000))
