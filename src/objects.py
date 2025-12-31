from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
import numpy as np
import os


class MPMModel(Enum):
    WATER = 0
    JELLY = 1
    SNOW = 2
    SAND = 3


@dataclass
class MPMMaterial:
    model: MPMModel = MPMModel.WATER
    density: float = 1000.0
    E: float = 1e5
    nu: float = 0.2
    viscosity: float = 0.0
    hardening: float = 0.0
    stiffness: float = 200.0
    power: float = 7.0


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
    collision_threshold: float = np.finfo(float).tiny
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scripted_trajectory: Optional[
        Callable[
            [float],
            Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
        ]
    ] = None
    is_dynamic: bool = True
    material: RigidMaterial = field(default_factory=RigidMaterial)


@dataclass
class MPMObject:
    meshdir: str
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    num_particles: int = 10000
    material: MPMMaterial = field(default_factory=MPMMaterial)


# ---------------- Cloth 部分 ----------------

@dataclass
class ClothMaterial:
    """
    布料材料参数：
    - k_struct: 结构弹簧（边 / 斜边）的刚度
    - k_bend: 弯曲弹簧刚度
    - damping: 弹簧阻尼（沿弹簧方向）
    - density: 面密度（这里主要给你留接口，目前 cloth.py 用的是每点平均质量）
    """
    k_struct: float = 2e4
    k_bend: float = 5e3
    damping: float = 50.0
    density: float = 1.0


@dataclass
class ClothObject:
    """
    单个布料对象的高层描述，供 Cloth 求解器使用。

    cloth.py 里目前假设：
        - obj.mesh: 已经加载好的 trimesh.Trimesh
        - obj.position: 初始平移
        - obj.velocity: 初始整体速度
        - obj.mass: 总质量（用于平均分配到各顶点）

    这里在 __post_init__ 里自动根据 meshdir 加载 mesh，
    和 RigidObject + RigidBody 的风格保持一致。
    """
    meshdir: str                      # .obj 文件或预定义几何名字
    position: Tuple[float, float, float]
    mass: float = 1.0                 # 总质量，会在 cloth.py 里均分到每个顶点
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 可选：事先指定要固定的局部顶点索引（local index）
    pinned_verts: Optional[List[int]] = None

    material: ClothMaterial = field(default_factory=ClothMaterial)

    # 运行时实际使用的 mesh，__post_init__ 中根据 meshdir 自动填充
    mesh: Optional["object"] = field(default=None, repr=False)

    def __post_init__(self):
        import trimesh
        from src.core.utils import GEOMS

        if self.mesh is not None:
            # 用户自己传了 mesh，就直接用
            return

        try:
            # 支持和刚体一样的“预定义几何”名字（比如以后你想做一个简单布面 primitive）
            if self.meshdir in GEOMS:
                mesh = GEOMS[self.meshdir](self.scale)
            else:
                # 尝试直接当路径加载
                mesh_path = self.meshdir
                if not os.path.isabs(mesh_path) and not os.path.exists(mesh_path):
                    # 常见情况：只写了文件名，默认从 meshes/ 目录找
                    alt = os.path.join("meshes", mesh_path)
                    if os.path.exists(alt):
                        mesh_path = alt
                mesh = trimesh.load(mesh_path, force="mesh")

            mesh.vertices = mesh.vertices.astype(np.float32)
            mesh.faces = mesh.faces.astype(np.int32)
        except Exception as e:
            raise RuntimeError(f"Failed to load cloth mesh '{self.meshdir}': {e}")

        self.mesh = mesh
