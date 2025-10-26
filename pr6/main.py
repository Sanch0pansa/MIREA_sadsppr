import numpy as np
import random
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

    def get_biggest_undiscovered_region(self) -> None:
        dimensions_by_coords = []
        for _ in self._agents[0]:
            dimensions_by_coords.append([])
        for agent in self._agents:
            for i, coord in enumerate(agent):
                dimensions_by_coords[i].append(coord)
        for dimension_list in dimensions_by_coords:
            dimension_list.sort()
        undiscovered_region_coords = np.zeros(len(dimensions_by_coords))
        for i, dimension_list in enumerate(dimensions_by_coords):
            biggest_distance = np.float64(0)
            last_value = dimension_list[0]
            for coord in dimension_list[1:]:
                if coord - last_value > biggest_distance:
                    biggest_distance = coord - last_value
                    undiscovered_region_coords[i] = (coord + last_value) / 2
                last_value = coord
        return undiscovered_region_coords

    def test_hypothesis(self) -> list[bool]:
        avg_value = np.average(
            np.array([self._function(agent) for agent in self._agents])
        )
        return [
            self._function(agent) <= avg_value
            for agent in self._agents
        ]

    def step(self) -> None:
        agents_activity = self.test_hypothesis()
        new_agents = self._agents.copy()
        distance = 0.2
        for (i, agent) in enumerate(self._agents):
            if not agents_activity[i]:
                j = random.randint(0, len(self._agents) - 2)
                if j == i:
                    j = len(self._agents) - 1
                if agents_activity[j]:
                    new_agents[i] = self._agents[j] + np.random.random(size=self._agents[j].shape) * distance - distance / 2
                else:
                    new_agents[i] = get_random_point_from_region(self._field_from, self._field_to)
        self._agents = new_agents
    
    def optimize(self) -> np.ndarray:
        self.generate_initial_hypothesis()
        stable_iterations = 0
        last_best_agent = self._agents[0]
        for i in range(2000):
            self.step()
            
            best_agent = min(self._agents, key=self._function)
            if (best_agent - last_best_agent > 0.005).any():
                stable_iterations = 0
            else:
                stable_iterations += 1
            if stable_iterations > 100:
                return best_agent
            last_best_agent = best_agent
            print(best_agent, self._function(best_agent))


def drop_wave_function(coords):
    x, y = coords[0], coords[1]
    return (
        -1 * (
            (1 + np.cos(12 * (
                np.sqrt(x ** 2 + y ** 2)
            )))
            /
            (
                0.5 * (x ** 2 + y ** 2) + 2
            )
        )
    )


if __name__ == "__main__":
    search = DiffusionSearch(
        function=drop_wave_function,
        field_from=np.array([-5.12, -5.12]),
        field_to=np.array([5.12, 5.12]),
        n_agents=1000,
    )

    print(search.optimize())
