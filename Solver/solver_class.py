from abc import ABC, abstractmethod
from dataclasses import dataclass
import networkx as nx
import numpy as np


@dataclass
class SolverResult:
    energy: float
    ordering: np.ndarray
    is_feasible: bool
    raw_sample: dict = None


class BaseSolver(ABC):
    
    @abstractmethod
    def configure(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def solve(self, graph: nx.Graph) -> SolverResult:
        pass
    
    @abstractmethod
    def _decode_solution(self, raw_sample: dict, num_nodes: int) -> tuple[np.ndarray, bool]:
        pass
    
    @abstractmethod
    def _build_qubo(self, graph: nx.Graph):
        pass
    
    @abstractmethod
    def _extract_Q_matrix(self, bqm) -> tuple[dict, float]:
        pass
