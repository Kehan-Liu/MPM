from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum


class MPMModel(Enum):
    WATER = 0
    JELLY = 1
    SNOW = 2


@dataclass
class MPMMaterial:
    model: MPMModel = MPMModel.WATER
    density: float = 1000.0
    E: float = 1e5
    nu: float = 0.2
    hardening: float = 0.0


@dataclass
class RigidMaterial:
    kh: float = 10
    friction: float = 0.5
    restitution: float = 0.0
    splitter: float = 0.0


@dataclass
class RigidObject:
    meshdir: str
    position: Tuple[float, float, float]
    mass: float
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scripted_trajectory: Optional[
        Callable[
            [float],
            Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        ]
    ] = None
    material: RigidMaterial = field(default_factory=RigidMaterial)


@dataclass
class MPMObject:
    meshdir: str
    position: Tuple[float, float, float]
    num_particles: int = 10000
    material: MPMMaterial = field(default_factory=MPMMaterial)


@dataclass
class ClothObject:
    pass
