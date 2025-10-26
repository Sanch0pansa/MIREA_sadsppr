import numpy as np
from collections.abc import Callable
from typing import TypeAlias

FunctionType: TypeAlias = Callable[[np.ndarray], np.floating]


def get_random_point_from_region(
    region_from: np.ndarray,
    region_to: np.ndarray,
) -> np.ndarray:
    return np.random.rand(region_from.shape[0]) * (region_to - region_from) + region_from


class DiffusionSearch:
    _function: FunctionType
    _field_from: np.ndarray
    _field_to: np.ndarray
    _n_agents: int
    _agents: list[np.ndarray]

    def __init__(
        self,
        function: FunctionType,
        field_from: np.ndarray,
        field_to: np.ndarray,
        n_agents: int,
    ) -> None:
        self._function = function
        self._field_from = field_from
        self._field_to = field_to
        self._n_agents = n_agents
        self._agents = []

    def generate_initial_hypothesis(self) -> None:
        self._agents = [
            get_random_point_from_region(self._field_from, self._field_to)
            for _ in range(self._n_agents)
        ]

    def compare_hypothesis(self) -> None:
        pass

    def step(self) -> None:
        self.generate_initial_hypothesis()
