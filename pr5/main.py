import numpy as np
from collections.abc import Callable
from typing import TypeAlias

FunctionType: TypeAlias = Callable[[np.ndarray], np.floating]


def get_random_point_from_region(
    region_from: np.ndarray,
    region_to: np.ndarray,
) -> np.ndarray:
    return np.random.rand(region_from.shape[0]) * (region_to - region_from) + region_from


def distance(
    point1: np.ndarray,
    point2: np.ndarray,
) -> np.floating:
    return np.linalg.norm(point1 - point2)


class BeeColony:
    _function: FunctionType
    _field_from: np.ndarray
    _field_to: np.ndarray

    _area_size: np.floating
    _area_intersection_distance: np.floating

    _n_scouts: int
    _n_bees_to_region: int

    def __init__(
        self,
        function: FunctionType,
        field_from: np.ndarray,
        field_to: np.ndarray,
        area_size: np.floating,
        area_intersection_distance: np.floating,
        n_scouts: int,
        n_bees_to_region: int,
    ) -> None:
        self._function = function
        self._field_from = field_from
        self._field_to = field_to
        self._area_size = area_size
        self._area_intersection_distance = area_intersection_distance
        self._n_scouts = n_scouts
        self._n_bees_to_region = n_bees_to_region

    def generate_scouts_regions_centers(self) -> list[np.ndarray]:
        return [
            get_random_point_from_region(
                region_from=self._field_from,
                region_to=self._field_to,
            ) for i in range(self._n_scouts)
        ]

    def get_unique_regions_centers(self, regions_centers: list[np.ndarray]) -> list[np.ndarray]:
        sorted_regions_centers = sorted(regions_centers, key=self._function)[:self._n_scouts]
        unique_sorted_regions_centers = []
        for i, region in enumerate(sorted_regions_centers):
            real_region = region
            for j in range(i-1, -1, -1):
                best_region = sorted_regions_centers[j]
                if distance(best_region, region) < self._area_intersection_distance:
                    real_region = region
            unique_sorted_regions_centers.append(real_region)
        
        return unique_sorted_regions_centers

    def get_scouts_regions_centers(self) -> list[np.ndarray]:
        regions_centers = self.generate_scouts_regions_centers()
        return self.get_unique_regions_centers(regions_centers)

    def get_region_from_center(self, region_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            region_center - self._area_size,
            region_center + self._area_size,
        )

    def generate_bees_in_region(self, region_center: np.ndarray, n_bees: int) -> list[np.ndarray]:
        region_from, region_to = self.get_region_from_center(region_center)
        return [
            get_random_point_from_region(
                region_to,
                region_from,
            ) for _ in range(n_bees)
        ]

    def step(self, regions_centers: list[np.ndarray]) -> list[np.ndarray]:
        regions_centers = self.get_unique_regions_centers(regions_centers)
        print("Скауты:")
        print(*regions_centers, sep="\n")

        new_region_centers = []
        for i, region_center in enumerate(regions_centers):
            bees = self.generate_bees_in_region(
                region_center=region_center,
                n_bees=self._n_bees_to_region,
            )
            print(f"\n- Пчелы в области {i} ({region_center}):")
            for bee in bees:
                print(bee, "функция:", self._function(bee))
            bees.append(region_center)
            bees.sort(key=self._function)
            if (np.abs(bees[0] - region_center) > 0.0001).all():
                print(f"Обновление области {i}: {bees[0]} ({self._function(bees[0])})")
            new_region_centers.append(bees[0])

        return new_region_centers

    def optimize(self, n_iterations: int) -> None:
        regions = self.generate_scouts_regions_centers()
        last_regions = regions
        for i in range(n_iterations):
            print(f"\n\n====== Шаг {i} ======")
            regions = self.step(regions)
            for region in regions:
                

         self.get_unique_regions_centers(regions)[0]


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


np.random.seed(1)
colony = BeeColony(
    function=drop_wave_function,
    field_from=np.array([-5.12, -5.12]),
    field_to=np.array([5.12, 5.12]),
    area_size=0.5,
    area_intersection_distance=0.5,
    n_scouts=5,
    n_bees_to_region=5,
)
res = colony.optimize(200)
print("\n\n")
print("Результат:", res)
print("Функция: ", drop_wave_function(res))
