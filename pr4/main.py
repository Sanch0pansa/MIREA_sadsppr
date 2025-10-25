from typing import TypeVar, Generic, TypeAlias
from abc import ABC, abstractmethod
from random import choices
from collections.abc import Callable
import math
import numpy as np
import json
from matplotlib import pyplot as plt

GraphType: TypeAlias = dict[str, dict[str, float]]


class Ant:
    def __init__(self, current_vertex: str) -> None:
        self._path = [current_vertex]
        self._current_vertex = current_vertex
        self._visited_vertices = {current_vertex}
        self._finished = False

    def choice_new_vertex(
        self,
        distances: GraphType,
        pheromones: GraphType,
        print_state: bool = False,
    ) -> None:
        alpha = 2
        beta = 2
        possible_vertices = set(distances[self._current_vertex].keys()) - self._visited_vertices
        sum_values = sum(
            (
                pheromones[self._current_vertex][vertex] ** alpha
                *
                (1 / distances[self._current_vertex][vertex]) ** beta
            ) for vertex in possible_vertices
        )
        probabilities = {
            vertex: (
                pheromones[self._current_vertex][vertex] ** alpha
                *
                (1 / distances[self._current_vertex][vertex]) ** beta
             ) / sum_values
            for vertex in possible_vertices
        }
        if print_state:
            print(self._path, probabilities)
        next_vertex = choices(list(probabilities.keys()), k=1, weights=list(probabilities.values()))[0]
        self._visited_vertices.add(next_vertex)
        self._path.append(next_vertex)
        self._current_vertex = next_vertex

        if len(possible_vertices) == 1:
            self._finished = True

    @property
    def current_vertex(self) -> str:
        return self._current_vertex

    @property
    def path(self) -> list[str]:
        return self._path

    def get_path_cost(
        self,
        distances: GraphType,
    ) -> None:
        path_cost = 0
        for i in range(1, len(self._path)):
            from_vertex = self._path[i - 1]
            to_vertex = self._path[i]
            path_cost += distances[from_vertex][to_vertex]
        path_cost += distances[self._path[-1]][self._path[0]]
        return path_cost
    
    def get_pheromones_increments(
        self,
        distances: GraphType,
    ) -> dict[str, float]:
        pheromones_increment = 1 / self.get_path_cost(distances=distances)
        edges = [
            (self._path[i - 1], self._path[i]) for i in range(1, len(self._path))
        ]
        return {edge: pheromones_increment for edge in edges}


class AntOptimization:
    _distances: GraphType
    _pheromones: GraphType

    def __init__(
        self,
        distances: GraphType,
    ):
        self._distances = distances
        self._pheromones = {
            i: {
                j: 0.001 for j in distances[i]
            } for i in distances
        }
    
    def step(self, n_ants: int, start_vertex: str, evaporation: float = 0.6, print_state: bool = False):
        for x in self._pheromones:
            for y in self._pheromones[x]:
                # print((x, y), self._pheromones[x][y])
                self._pheromones[x][y] *= (1 - evaporation)
                self._pheromones[x][y] = max(self._pheromones[x][y], 0.001)

        ants = [Ant(start_vertex) for _ in range(n_ants)]
        for _ in range(len(self._distances) - 1):
            for i, ant in enumerate(ants):
                ant.choice_new_vertex(self._distances, self._pheromones, False)

        pathes = {}
        pathes_counter = {}
        for i, ant in enumerate(ants):
            pathes[tuple(ant.path)] = ant.get_path_cost(self._distances)
            pathes_counter[tuple(ant.path)] = pathes_counter.get(tuple(ant.path), 0) + 1
        
        shortest_path = max(pathes, key=pathes.get)
        if print_state:
            for i, ant in enumerate(ants):
                print(f"Ant {i}: {ant.path}", ant.get_path_cost(self._distances))

        for ant in ants:
            pheromones_increments = ant.get_pheromones_increments(distances=self._distances)
            for from_vertex, to_vertex in pheromones_increments:
                self._pheromones[from_vertex][to_vertex] += pheromones_increments[(from_vertex, to_vertex)]
        print("Самый короткий маршрут: ", shortest_path, pathes[shortest_path])
        print("Все расмотренные пути:")
        for path in pathes_counter:
            print(path, "Муравьев:", pathes_counter[path])
        

def traveling():
    with open("pr2/addresses_paths.json") as f:
        distances = json.loads(f.read())
    opt = AntOptimization(distances)
    for i in range(250):
        print(f"=== Шаг {i} ===")
        opt.step(15, "Дом", 0.6, False)
    

traveling()