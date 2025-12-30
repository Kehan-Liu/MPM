from dataclasses import dataclass, field
from typing import List, Tuple
from src.objects import RigidObject, MPMObject, ClothObject

@dataclass
class Scene:
    rigid_objects: List[RigidObject] = field(default_factory=list)
    cloth_objects: List[ClothObject] = field(default_factory=list)
    mpm_objects: List[MPMObject] = field(default_factory=list)
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    n_grid: int = 128
    dt: float = 1e-4
    penalty_parameter: float = 1
    clamp_factor: float = 20.0
    

    def add_rigid_object(self, rigid_object: RigidObject):
        self.rigid_objects.append(rigid_object)
    def add_cloth_object(self, cloth_object: ClothObject):
        self.cloth_objects.append(cloth_object)
    def add_mpm_object(self, mpm_object: MPMObject):
        self.mpm_objects.append(mpm_object)