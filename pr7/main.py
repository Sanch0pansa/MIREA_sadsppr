import numpy as np
import random
from collections.abc import Callable
from typing import TypeAlias
from matplotlib import pyplot as plt

FunctionType: TypeAlias = Callable[[np.ndarray], np.floating]


def get_random_point_from_region(
    region_from: np.ndarray,
    region_to: np.ndarray,
) -> np.ndarray:
    return np.random.rand(region_from.shape[0]) * (region_to - region_from) + region_from


class GeneticAlgorithmOptimization:
    _function: FunctionType
    _field_from: np.ndarray
    _field_to: np.ndarray
    _n_agents: int
    _survival_coefficient: np.floating
    _mutation_coefficient: np.floating
    _mutation_normal_distribution: np.floating

    _agents: list[np.ndarray]

    def __init__(
        self,
        function: FunctionType,
        n_agents: int,
        survival_coefficient: np.floating,
        mutation_coefficient: np.floating,
        mutation_normal_distribution: np.floating,
        field_from: np.ndarray,
        field_to: np.ndarray,
    ) -> None:
        self._function = function
        self._n_agents = n_agents
        self._survival_coefficient = survival_coefficient
        self._mutation_coefficient = mutation_coefficient
        self._field_from = field_from
        self._field_to = field_to
        self._mutation_normal_distribution = mutation_normal_distribution

        self._agents = [
            get_random_point_from_region(self._field_from, self._field_to)
            for _ in range(self._n_agents)
        ]
    
    def get_default_agents_fitness(self) -> list[np.floating]:
        return [self._function(agent) for agent in self._agents]

    def get_rank_agents_fitness(self) -> list[np.floating]:
        fitnesses = [self._function(agent) for agent in self._agents]
        indices = list(range(self._n_agents))
        sorted_indices = sorted(indices, key=lambda x: fitnesses[x])
        ranked_fitnesses = [0 for _ in range(self._n_agents)]
        for i in range(self._n_agents):
            ranked_fitnesses[sorted_indices[i]] = np.float64(i + 1)
        return ranked_fitnesses

    def get_agents_fitness(self) -> list[np.floating]:
        return self.get_rank_agents_fitness()

    def select_random_agents(
        self,
        agents_fitness: list[np.floating],
    ) -> list[np.ndarray]:
        sum_fitness = np.sum(agents_fitness)
        indices = np.random.choice(
            a=self._n_agents,
            size=int(self._n_agents * self._survival_coefficient),
            replace=False,
            p=np.array(agents_fitness) / sum_fitness,
        )
        return [self._agents[idx] for idx in indices]
    
    def select_best_agents(
        self,
        agents_fitness: list[np.floating],
    ) -> list[np.ndarray]:
        agents_to_select = int(self._n_agents * self._survival_coefficient)
        indices = list(range(self._n_agents))
        sorted_indices = sorted(indices, key=lambda x: -agents_fitness[x])
        selected_indices = sorted_indices[:agents_to_select]
        return [self._agents[idx] for idx in selected_indices]

    def select_agents(
        self,
        agents_fitness: list[np.floating],
    ) -> list[np.ndarray]:
        return self.select_best_agents(
            agents_fitness=agents_fitness,
        )

    def make_pairs(
        self,
        selected_agents: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_pairs = self._n_agents // 2
        pairs = []
        for i in range(n_pairs):
            pair_indices = np.random.choice(
                len(selected_agents),
                size=2,
                replace=False,
            )
            pairs.append(
                (
                    selected_agents[pair_indices[0]],
                    selected_agents[pair_indices[1]],
                )
            )
        
        return pairs

    def interpolation_crossover(
        self,
        agent1: np.ndarray,
        agent2: np.ndarray,
    ) -> np.ndarray:
        alpha = np.random.random(size=agent1.shape[0])
        new_agent = (
            agent1 * alpha +
            agent2 * (1 - alpha)
        )
        return new_agent

    def crossover_pair(
        self,
        agent1: np.ndarray,
        agent2: np.ndarray,
    ) -> np.ndarray:
        return self.interpolation_crossover(agent1, agent2)

    def make_crossover(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray]:
        new_generation = []
        for pair in pairs:
            new_generation.append(self.crossover_pair(*pair))
            new_generation.append(self.crossover_pair(*pair))

        return new_generation
    
    def normal_mutate_agent(
        self,
        agent: np.ndarray,
    ) -> np.ndarray:
        return agent + np.random.randn(agent.shape[0]) * self._mutation_normal_distribution

    def mutate_agent(
        self,
        agent: np.ndarray,
    ) -> np.ndarray:
        return self.normal_mutate_agent(agent)

    def make_mutations(
        self,
        new_generation: list[np.ndarray],
    ) -> list[np.ndarray]:
        agents_to_mutate = int(self._n_agents * self._mutation_coefficient)
        for _ in range(agents_to_mutate):
            agent_idx = np.random.randint(0, self._n_agents - 1)
            new_generation[agent_idx] = self.mutate_agent(new_generation[agent_idx])

    def get_best_solution(
        self,
    ) -> np.ndarray:
        return max(self._agents, key=self._function)

    def coord_to_str(
        self,
        coord: np.ndarray,
    ) -> str:
        return f"({coord[0].round(4)}; {coord[1].round(4)})"

    def print_agents(self) -> None:
        print("Агенты:")
        for i, agent in enumerate(self._agents):
            print(f"- X{i + 1} = {self.coord_to_str(agent)}")

    def print_agents_functions(
        self,
    ) -> None:
        print("Значения целевых функций:")
        for i, agent in enumerate(self._agents):
            print(f"- f(X{i + 1}) = {-self._function(agent).round(4)}")

    def print_agents_fitness(
        self,
        agents_fitness: list[np.floating],
    ) -> None:
        print("Значения фитнесс-функций:")
        for i, agent_fitness in enumerate(agents_fitness):
            print(f"- f(X{i + 1}) = {agent_fitness.round(4)}")

    def print_selected_agents(
        self,
        selected_agents: list[np.ndarray],
    ) -> None:
        print("Выбранные агенты:")
        for i, agent in enumerate(selected_agents):
            print(f"- X{i + 1} = {self.coord_to_str(agent)}")
    
    def print_pairs(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        print("Выбранные пары:")
        for i, j in pairs:
            print(f"- {self.coord_to_str(i)} и {self.coord_to_str(j)}")

    def print_new_generation(
        self,
        new_generation: list[np.ndarray],
    ) -> None:
        print("Новое поколение агентов:")
        for i, agent in enumerate(new_generation):
            print(f"- X{i + 1} = {self.coord_to_str(agent)}")
    
    def print_new_generation_with_mutations(
        self,
        new_generation: list[np.ndarray],
    ) -> None:
        print("Новое поколение агентов с мутациями:")
        for i, agent in enumerate(new_generation):
            print(f"- X{i + 1} = {self.coord_to_str(agent)}")

    def print_best_solution(
        self,
    ) -> None:
        best_solution = self.get_best_solution()
        print("Лучшее решение:", best_solution)
        print("Лучшее значение функции:", -self._function(best_solution))

    def step(
        self,
        print_state: bool = False
    ) -> None:
        if print_state:
            self.print_agents()
            self.print_agents_functions()

        agents_fitness = self.get_agents_fitness()
        if print_state:
            self.print_agents_fitness(agents_fitness)

        selected_agents = self.select_agents(agents_fitness=agents_fitness)
        if print_state:
            self.print_selected_agents(selected_agents)

        pairs = self.make_pairs(selected_agents=selected_agents)
        if print_state:
            self.print_pairs(pairs)

        new_generation = self.make_crossover(pairs=pairs)
        if print_state:
            self.print_new_generation(new_generation)

        self.make_mutations(new_generation)
        if print_state:
            self.print_new_generation_with_mutations(new_generation)

        self._agents = new_generation
        self.print_best_solution()
    
    def start_visualization(self) -> None:
        plt.xlim(self._field_from[0], self._field_to[0])
        plt.ylim(self._field_from[1], self._field_to[1])

    def visualize_agents(
        self, color,
    ) -> None:
        plt.scatter(
            x=[agent[0] for agent in self._agents],
            y=[agent[1] for agent in self._agents],
            c=color,
            s=10,
        )

    def visualize_step(
        self,
        current_iteration: int,
        max_iteration: int,
    ) -> None:
        x = (current_iteration / max_iteration)
        color = (x, 0, 1-x)
        self.visualize_agents(color)

    def optimize(
        self,
        n_iterations: int,
        print_state: bool = False,
    ) -> np.ndarray:
        self.start_visualization()
        for i in range(n_iterations):
            print(f"===== Шаг {i + 1} =====")
            self.step(print_state)
            if i in [0, n_iterations-1]:
                self.visualize_step(
                    current_iteration=i,
                    max_iteration=n_iterations - 1,
                )


def drop_wave_function(coords):
    x, y = coords[0], coords[1]
    return (
        1 * (
            (1 + np.cos(12 * (
                np.sqrt(x ** 2 + y ** 2)
            )))
            /
            (
                0.5 * (x ** 2 + y ** 2) + 2
            )
        )
    )


np.random.seed(1)
generic_algorithm = GeneticAlgorithmOptimization(
    function=drop_wave_function,
    survival_coefficient=0.5,
    mutation_coefficient=0.25,
    field_from=np.array([-5.12, -5.12]),
    field_to=np.array([5.12, 5.12]),
    n_agents=1000,
    mutation_normal_distribution=0.5,
)

result = generic_algorithm.optimize(150, print_state=False)
generic_algorithm.print_best_solution()
# plt.show()