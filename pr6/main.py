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
    _max_distance_from_active_agent: np.floating

    def __init__(
        self,
        function: FunctionType,
        field_from: np.ndarray,
        field_to: np.ndarray,
        n_agents: int,
        max_distance_from_active_agent: np.floating
    ) -> None:
        self._function = function
        self._field_from = field_from
        self._field_to = field_to
        self._n_agents = n_agents
        self._agents = []
        self._max_distance_from_active_agent = max_distance_from_active_agent

    def generate_initial_hypothesis(self) -> None:
        self._agents = [
            get_random_point_from_region(self._field_from, self._field_to)
            for _ in range(self._n_agents)
        ]

    def get_agents_activity_by_average_function_value(self) -> list[bool]:
        avg_value = np.average(
            np.array([self._function(agent) for agent in self._agents])
        )
        return [
            self._function(agent) <= avg_value
            for agent in self._agents
        ]
    
    def get_random_point_close_to_active_agent(
        self,
        active_agent: np.ndarray,
    ) -> np.ndarray:
        return active_agent + (
            np.random.random(size=active_agent.shape) * self._max_distance_from_active_agent
            - self._max_distance_from_active_agent / 2
        )
    
    def print_state(self) -> None:
        agents_activity = self.get_agents_activity_by_average_function_value()
        avg_value = np.average(
            np.array([self._function(agent) for agent in self._agents])
        )
        print("Среднее значение функции:", avg_value.round(4))
        print("Размещение агентов:")
        for i, (agent, agent_activity) in enumerate(zip(self._agents, agents_activity)):
            print(f"x{i + 1}(0) = ({agent[0].round(4)}; {agent[1].round(4)}); f(x{i + 1}) = {self._function(agent).round(4)}; a{i + 1}={int(agent_activity)}")

    def print_iteration_for_inactive_agent(
        self,
        i: int,
        j: int,
        agents_activity: list[bool],
        new_agent_position: np.ndarray,
    ) -> None:
        print(f"Агент {i + 1}. Выбор агента {j + 1}.")
        activity = "активен" if agents_activity[j] else "неактивен"
        print(f"- агент {j + 1} {activity}")
        print(f"- новая позиция агента {i+1}: ({new_agent_position[0].round(4)}; {new_agent_position[1].round(4)})")

    def step(self) -> None:
        agents_activity = self.get_agents_activity_by_average_function_value()
        new_agents = self._agents.copy()
        # self.print_state()
        for i in range(self._n_agents):
            if not agents_activity[i]:
                j = random.randint(0, len(self._agents) - 2)
                if j == i:
                    j = len(self._agents) - 1
                if agents_activity[j]:
                    new_agents[i] = self.get_random_point_close_to_active_agent(self._agents[j])
                else:
                    new_agents[i] = get_random_point_from_region(self._field_from, self._field_to)
                # self.print_iteration_for_inactive_agent(
                #     i,
                #     j,
                #     agents_activity,
                #     new_agents[i],
                # )
        self._agents = new_agents

    def print_current_iteration_result(
        self,
        iteration_number: int,
        best_solution: np.ndarray,
        best_value: np.floating,
    ) -> None:
        print(f"===== Шаг {iteration_number + 1} =====")
        print(f"Лучшее решение: {best_solution}")
        print(f"Лучшее значение функции: {best_value}")

    def optimize(
        self,
        n_iterations: int = 1_000_000,
        stop_after_n_stable_iterations: int = 25,
    ) -> np.ndarray:
        self.generate_initial_hypothesis()
        stable_iterations = 0
        last_best_agent = self._agents[0]
        for i in range(n_iterations):
            self.step()
            best_agent = min(self._agents, key=self._function)
            if (best_agent - last_best_agent > 0.005).any():
                stable_iterations = 0
            else:
                stable_iterations += 1
            if stable_iterations > stop_after_n_stable_iterations:
                return best_agent
            last_best_agent = best_agent
            self.print_current_iteration_result(
                i,
                best_agent,
                self._function(best_agent),
            )
        return last_best_agent


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
    np.random.seed(1)
    random.seed(1)
    search = DiffusionSearch(
        function=drop_wave_function,
        field_from=np.array([-5.12, -5.12]),
        field_to=np.array([5.12, 5.12]),
        n_agents=1000,
        max_distance_from_active_agent=0.2,
    )

    print(search.optimize())
