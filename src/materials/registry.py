from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Sequence, Tuple


class MaterialCategory(IntEnum):
    ELASTIC = 0
    FLUID = 1
    CLOTH = 2


@dataclass
class MaterialConfig:
    name: str
    density: float
    youngs_modulus: float
    poisson_ratio: float
    hardening: float = 10.0
    viscosity: float = 0.0
    category: MaterialCategory = MaterialCategory.ELASTIC
    color: Tuple[float, float, float] = (0.9, 0.9, 0.9)


class MaterialRegistry:
    """Holds host-side material definitions before being uploaded to Taichi."""

    def __init__(self) -> None:
        self._materials: List[MaterialConfig] = []
        self._name_to_id: Dict[str, int] = {}

    def register(self, config: MaterialConfig) -> int:
        if config.name in self._name_to_id:
            raise ValueError(f"Material '{config.name}' already registered")
        material_id = len(self._materials)
        self._materials.append(config)
        self._name_to_id[config.name] = material_id
        return material_id

    def __len__(self) -> int:
        return len(self._materials)

    def __iter__(self):
        return iter(self._materials)

    def get(self, name_or_id: int | str) -> MaterialConfig:
        if isinstance(name_or_id, str):
            return self._materials[self._name_to_id[name_or_id]]
        return self._materials[name_or_id]

    def to_solver_payload(self) -> Dict[str, Sequence[float]]:
        if not self._materials:
            raise RuntimeError(
                "Register at least one material before building the solver"
            )
        densities: List[float] = []
        mu_values: List[float] = []
        lambda_values: List[float] = []
        bulk_values: List[float] = []
        hardening: List[float] = []
        viscosities: List[float] = []
        categories: List[int] = []
        colors: List[Tuple[float, float, float]] = []
        for mat in self._materials:
            E = mat.youngs_modulus
            nu = mat.poisson_ratio
            mu = E / (2.0 * (1.0 + nu))
            _lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            bulk = E / (3.0 * (1.0 - 2.0 * nu))
            densities.append(mat.density)
            mu_values.append(mu)
            lambda_values.append(_lambda)
            bulk_values.append(bulk)
            hardening.append(mat.hardening)
            viscosities.append(mat.viscosity)
            categories.append(int(mat.category))
            colors.append(mat.color)
        return {
            "density": densities,
            "mu": mu_values,
            "lambda": lambda_values,
            "bulk": bulk_values,
            "hardening": hardening,
            "viscosity": viscosities,
            "category": categories,
            "color": colors,
        }
