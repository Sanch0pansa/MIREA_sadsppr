from typing import TypeVar, Generic, TypeAlias
from abc import ABC, abstractmethod
from random import randint, sample
from collections.abc import Callable
import math
import numpy as np
import json
from matplotlib import pyplot as plt


SolutionType = TypeVar("SolutionType")


class Annealing(ABC, Generic[SolutionType]):
    temp: float
    solution: SolutionType
    min_energy: float = float('+inf')
    best_solution: SolutionType
    temp: float
    iterations: int = 1

    @abstractmethod
    def energy(self, solution: SolutionType) -> float:
        pass

    @abstractmethod
    def get_next_solution(self, solution: SolutionType) -> SolutionType:
        pass
   
    @abstractmethod
    def get_probability(self, delta_energy: float):
        pass

    @abstractmethod
    def decrease_temp(self) -> None:
        pass

    def get_true_with_prob(self, prob: float) -> bool:
        n = randint(1, 10_000) / 10_000
        return n < prob
    
    @abstractmethod
    def text_solution(self, solution: SolutionType) -> str:
        pass

    def step(self) -> None:
        print("Текущее решение:", self.text_solution(self.solution))
        print("Энергия текущего решения:", self.energy(self.solution))
        print("Текущая температура:", self.temp)
        next_solution = self.get_next_solution(self.solution)
        next_step_energy = self.energy(next_solution)
        current_step_energy = self.energy(self.solution)
        print("Рабочее решение:", self.text_solution(next_solution))
        print("Энергия рабочего решения:", next_step_energy)
        delta_energy = next_step_energy - current_step_energy
        print("Разница в энергии:", delta_energy)
        if delta_energy > 0:
            prob = self.get_probability(delta_energy)
            print("Вероятность перехода:", prob)
        if delta_energy < 0:
            print("Переходим в новое состояние")
            self.solution = next_solution
            if next_step_energy <= self.min_energy:
                print("Энергия меньше текущего минимума, обновляем")
                self.min_energy = next_step_energy
                self.best_solution = next_solution
        elif self.get_true_with_prob(self.get_probability(delta_energy)):
            self.solution = next_solution
            print("Решение принято")
        else:
            print("Решение не принято")
        self.iterations += 1
        self.decrease_temp()

    def print_best_solution(self) -> None:
        print("Лучшее решение:", self.text_solution(self.best_solution))
        print("Энергия системы:", self.min_energy)


class ExpAnnealing(Annealing[SolutionType], Generic[SolutionType]):
    def get_probability(self, delta_energy: float) -> float:
        return math.exp(-delta_energy / self.temp)


TravelingSolution: TypeAlias = list[str]

class TravelingAnnealing(ExpAnnealing[TravelingSolution]):

    def __init__(
        self,
        points: list[str],
        distances: dict[str, dict[str, float]],
        solution: list[str],
    ) -> None:
        self.points = points
        self.solution = solution
        self.distances = distances
        self.min_energy = self.energy(self.solution)
        self.best_solution = self.solution

        self.temp = 100

    def dist_between(self, point_from: str, point_to: str) -> float:
        return self.distances[point_from][point_to]

    def energy(self, solution: TravelingSolution) -> float:
        return sum(
            self.dist_between(solution[i], solution[i + 1])
            for i in range(len(solution) - 1)
        ) + self.dist_between(solution[-1], solution[0])

    def get_next_solution(self, solution: TravelingSolution) -> TravelingSolution:
        p1, p2 = sample(self.solution[1:], 2)
        replacements = {
            p1: p2,
            p2: p1,
        }
        new_solution = solution.copy()
        for i in range(len(new_solution)):
            new_solution[i] = replacements.get(solution[i], solution[i])
        return new_solution

    def decrease_temp(self) -> None:
        self.temp *= 0.995

    def text_solution(self, solution: TravelingSolution) -> str:
        return " -> ".join(solution + [solution[0]])


CoordsSolution: TypeAlias = tuple[float, float]


class FunctionAnnealing(ExpAnnealing[CoordsSolution]):
    def __init__(
        self,
        start_solution: CoordsSolution,
        function: Callable[[CoordsSolution], float],
    ) -> None:
        self.solution = start_solution
        self.function = function
        self.min_energy = self.energy(self.solution)
        self.best_solution = self.solution

        self.temp = 1
        self.max_temp = self.temp

    def energy(self, solution: TravelingSolution) -> float:
        return self.function(solution)

    def get_shift(self) -> float:
        u = randint(0, 10_000) / 10_000
        return (self.temp / self.max_temp) * math.tan(math.pi * (u - 0.5))
        # return u - 0.5

    def get_next_solution(self, solution: TravelingSolution) -> TravelingSolution:
        x, y = solution
        dx, dy = self.get_shift(), self.get_shift()
        return (x + dx, y + dy)

    def decrease_temp(self) -> None:
        self.temp = self.max_temp / (self.iterations ** (1/2))

    def text_solution(self, solution: TravelingSolution) -> str:
        x, y = solution

        return f"({int(x * 10000) / 10000}, {int(y * 10000) / 10000})"


def traveling():
    with open("pr2/addresses_paths.json") as f:
        distances = json.loads(f.read())

    tr = TravelingAnnealing(
        points=list(distances.keys()),
        distances=distances,
        solution=[
            "Дом",
            "ПНД",
            "ЕО",
            "ДД",
            "ДКЦ",
            "ТД",
            "НД",
        ]
    )

    for i in range(10000):
        print(f"\n==== ШАГ {i + 1} ====")
        tr.step()

    print("\n===== ИТОГ =====")
    tr.print_best_solution()


def function_optimization():

    values = []

    def drop_wave_function(coords):
        x, y = coords
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

    tr = FunctionAnnealing(
        start_solution=(1.56, -3.23),
        function=drop_wave_function,
    )


    for i in range(2500):
        print(f"\n==== ШАГ {i + 1} ====")
        values.append(tr.solution)
        tr.step()

    r = 10

    plt.xlim(-r, r)
    plt.ylim(-r, r)

    x = np.linspace(-r, r, 200)
    y = np.linspace(-r, r, 200)
    X, Y = np.meshgrid(x, y)
    Z = drop_wave_function((X, Y))

    plt.pcolormesh(X, Y, Z, shading='auto', cmap='plasma')

    plt.plot([v[0] for v in values], [v[1] for v in values], marker='o')

    print("\n===== ИТОГ =====")
    tr.print_best_solution()
    plt.show()

# function_optimization()
traveling()