import numpy as np
from collections.abc import Callable
from typing import TypeAlias
from abc import ABC, abstractmethod

FunctionType: TypeAlias = Callable[[np.ndarray], np.floating]


class Particle:
    __coords: np.ndarray
    __velocity: np.ndarray
    __best_solution_coords: np.ndarray
    __global_best_solution_coords: np.ndarray
    __coords_dimension: int
    __function: FunctionType

    def __init__(
        self,
        coords: np.ndarray,
        velocity: np.ndarray,
        function: FunctionType,
    ) -> None:
        self.__coords = coords
        self.__velocity = velocity
        self.__best_solution_coords = coords.copy()
        self.__global_best_solution_coords = coords.copy()
        self.__coords_dimension = coords.shape[0]
        self.__function = function
    
    def move(self) -> None:
        self.__coords += self.__velocity
        if self.value < self.best_value:
            self.__best_solution_coords = self.__coords.copy()
            if self.value < self.global_best_value:
                print("Перезаписываем глобальный минимум", self.value, self.global_best_value)
                self.__global_best_solution_coords = self.__coords.copy()

    def accelerate(
        self,
        cognitive_coefficient: np.floating,
        social_coefficient: np.floating,
        inertia: np.floating,
    ) -> None:
        vector_to_self_best_solution = (
            self.__best_solution_coords - self.__coords
        )
        vector_to_global_best_solution = (
            self.__global_best_solution_coords - self.__coords
        )
        r1 = np.random.rand(self.__coords_dimension)
        r2 = np.random.rand(self.__coords_dimension)
        self.__velocity = (
            self.__velocity * inertia +
            vector_to_self_best_solution * r1 * cognitive_coefficient +
            vector_to_global_best_solution * r2 * social_coefficient
        )
    
    @property
    def value(self) -> np.floating:
        return self.__function(self.__coords)

    @property
    def best_value(self) -> np.floating:
        return self.__function(self.__best_solution_coords)

    @property
    def global_best_value(self) -> np.floating:
        return self.__function(self.__global_best_solution_coords)

    def set_best_solution_coords(self, coords: np.ndarray) -> None:
        self.__global_best_solution_coords = coords.copy()
    
    @property
    def global_best_solution_coords(self) -> np.ndarray:
        return self.__global_best_solution_coords

    def print_state(self) -> None:
        # print(f"Координаты: {self.__coords}")
        # print(f"Собственное лучшее решение: {self.__best_solution_coords}; {self.__function(self.__best_solution_coords)}")
        # print(f"Глобальное лучшее решение: {self.__global_best_solution_coords}; {self.__function(self.__global_best_solution_coords)}")
        print(f"Координаты: ({self.__coords[0]}, {self.__coords[1]}); Скорость: ({self.__velocity[0]}, {self.__velocity[1]}); Лучшее значение функции: {self.best_value}.")


def build_particle(
    function: FunctionType,
    fields_from: np.ndarray,
    fields_to: np.ndarray,
) -> Particle:
    coords = (
        np.random.rand(*fields_from.shape) *
        (fields_to - fields_from) + fields_from
    )

    return Particle(
        coords,
        np.random.rand(*coords.shape),
        function,
    )


class Swarm(ABC):
    _particles: list[Particle]
    _n_particles: int
    _function: FunctionType
    _cognitive_coefficient: np.floating
    _social_coefficient: np.floating
    _inertia: np.floating

    def __init__(
        self,
        function: FunctionType,
        fields_from: np.ndarray,
        fields_to: np.ndarray,
        n_particles: int,
        cognitive_coefficient: np.floating,
        social_coefficient: np.floating,
        inertia: np.floating,
    ) -> None:
        self._n_particles = n_particles
        self._particles = [
            build_particle(function, fields_from, fields_to)
            for _ in range(n_particles)
        ]
        self._function = function
        self._cognitive_coefficient = cognitive_coefficient
        self._social_coefficient = social_coefficient
        self._inertia = inertia
    
    def step(self):
        for i, particle in enumerate(self._particles):
            print(f"{i} частица. ", end="")
            particle.print_state()
            particle.accelerate(
                cognitive_coefficient=self._cognitive_coefficient,
                social_coefficient=self._social_coefficient,
                inertia=self._inertia,
            )
            particle.move()
        
        self.update_particles_best_solutions()
    
    def find_global_best_solution_coords(self) -> np.ndarray:
        best_value = np.inf
        best_solution_coords = None
        for particle in self._particles:
            if particle.global_best_value < best_value:
                print("Нашли новое наименьшее решение:", particle.global_best_value)
                best_value = particle.global_best_value
                best_solution_coords = particle.global_best_solution_coords
        
        return best_solution_coords

    def optimize(self, n_steps: int) -> np.ndarray:
        for step in range(n_steps):
            self.step()
            coords = self.find_global_best_solution_coords()
            value = self._function(coords)

            print(f"[{step + 1}/{n_steps}] Coords: {coords}; value={value}")
        
        return self.find_global_best_solution_coords()
    
    @abstractmethod
    def update_particles_best_solutions(self) -> None:
        pass


class GlobalSwarm(Swarm):
    def update_particles_best_solutions(self) -> None:
        best_solution_coords = self.find_global_best_solution_coords()
        for particle in self._particles:
            particle.set_best_solution_coords(
                best_solution_coords,
            )

class LocalSwarm(Swarm):
    _neighbors: int

    def __init__(
        self,
        function: FunctionType,
        fields_from: np.ndarray,
        fields_to: np.ndarray,
        n_particles: int,
        cognitive_coefficient: np.floating,
        social_coefficient: np.floating,
        inertia: np.floating,
        neighbors: int,
    ) -> None:
        super().__init__(
            function=function,
            fields_from=fields_from,
            fields_to=fields_to,
            n_particles=n_particles,
            cognitive_coefficient=cognitive_coefficient,
            social_coefficient=social_coefficient,
            inertia=inertia,
        )
        self._neighbors = neighbors
    
    def update_particles_best_solutions(self) -> None:
        new_global_minimums = [
            particle.global_best_solution_coords.copy()
            for particle in self._particles
        ]
        for i in range(self._n_particles):
            best_value = np.inf
            best_solution_coords = None
            for j in range(i - self._neighbors, i + self._neighbors + 1):
                idx = j % self._n_particles
                particle = self._particles[idx]
                if particle.global_best_value < best_value:
                    best_value = particle.global_best_value
                    best_solution_coords = particle.global_best_solution_coords
            new_global_minimums[i] = best_solution_coords
        
        for i in range(self._n_particles):
            self._particles[i].set_best_solution_coords(new_global_minimums[i])
