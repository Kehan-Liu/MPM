Repository Structure
MPM/
│
├── core/                     # 基础核心工具（数学、数据结构、物理常量）
│   ├── math_utils.py
│   ├── interpolation.py
│   └── geometry.py
│
├── simulators/               # 三大 Solver + Hybrid 调度
│   ├── hybrid_simulator.py   # HybridSimulator（顶层调度器）
│   │
│   ├── solid/
│   │   ├── solid_object.py
│   │   ├── solid_solver.py
│   │   └── solid_material.py
│   │
│   ├── liquid/
│   │   ├── liquid_object.py
│   │   ├── liquid_solver.py
│   │   ├── mpm/              # =MPM
│   │      ├── particles.py
│   │      ├── grid.py
│   │      ├── constitutive_model.py
│   │      └── shape_functions.py
│   │
│   ├── cloth/
│   │   ├── cloth_object.py
│   │   ├── cloth_solver.py
│   │   └── cloth_constraints.py
│   │
│   └── coupling/             # 三种 pairwise coupling
│       ├── solid_liquid_coupling.py
│       ├── liquid_cloth_coupling.py
│       └── solid_cloth_coupling.py
│
├── collision/
│   ├── collider.py           # 抽象类
│   ├── plane_collider.py
│   ├── sphere_collider.py
│   ├── box_collider.py
│   └── sdf_collider.py       # （可选）SDF rigid body boundary
│
├── rendering/
│   ├── renderer.py
│   ├── solid_visualizer.py
│   ├── liquid_visualizer.py
│   └── cloth_visualizer.py
│
├── scenes/                   # 场景配置（如 json / python）
│   ├── scene1.py
│   ├── scene2_softbody.py
│   ├── scene3_cloth_fluid.py
│   └── scene4_all.py
│
├── configs/                  # 参数文件
│   ├── solid_params.yaml
│   ├── liquid_params.yaml
│   ├── cloth_params.yaml
│   └── scene_config.yaml
│
├── tests/                    # 单元测试
│   ├── test_solid.py
│   ├── test_liquid.py
│   ├── test_cloth.py
│   ├── test_coupling.py
│   └── test_collider.py
│
└── main.py                   # 程序入口，运行 HybridSimulator




整个设计的结构是
HybridSimulator
  - SolidSolver
  - LiquidSolver
  - ClothSolver
    - SolidLiquidCoupler
    - LiquidClothCoupler
    - SolidClothCoupler

在Solver调用step的过程中使用coupler进行多物理场耦合计算。
所有函数的第一行代表输入,第二行(如果有)代表输出.
# HybridSimulator
**外部接口**
- add_solid 向场景中添加一个固体对象
  - solid_obj: SolidObject
- add_liquid 向场景中添加一个液体对象
  - liquid_obj: LiquidObject
- add_cloth 向场景中添加一个布料对象
  - cloth_obj: ClothObject
- add_collider 向场景中添加一个碰撞体
  - collider: Collider
- add_solid_liquid_coupling 添加固体-液体耦合器
  - coupler: SolidLiquidCoupling
- add_liquid_cloth_coupling 添加液体-布料耦合器
  - coupler: LiquidClothCoupling    
- add_solid_cloth_coupling 添加固体-布料耦合器
  - coupler: SolidClothCoupling
- step: 执行一次物理时间步。顺序为：solid 解算 liquid 解算 cloth 解算 各自内部碰撞（self collision） 三种 cross coupling 交互 环境碰撞体处理 渲染（可选）
  - dt: 时间步长, render: 是否渲染当前帧
- run: 运行多个物理时间步
  - num_steps: 时间步数量, dt: 时间步长, render: 是否渲染每一帧

**内部接口**
- _step_solid: 执行固体解算器的时间步
  - dt: 时间步长
- _step_liquid: 执行液体解算器的时间步
  - dt: 时间步长
- _step_cloth: 执行布料解算器的时间步
  - dt: 时间步长
- _apply_solid_liquid_coupling: 执行固体-液体耦合计算
  - dt: 时间步长
- _apply_liquid_cloth_coupling: 执行液体-布料耦合计算
  - dt: 时间步长
- _apply_solid_cloth_coupling: 执行固体-布料耦合计算
  - dt: 时间步长
- _render: 渲染当前场景状态
# Solid Module
## SolidObject
- get_state: 获取当前固体对象的状态（位置、速度、应变等）
  - None
  - state: dict
- set_state: 设置当前固体对象的状态
  - state: dict
- apply_external_forces: 加入重力或者外力.
  - dt, global_forces: dict or None
## SolidSolver
管理所有SolidObject对象并执行时间步
**外部接口**
- add_solid 添加一个固体对象
  - solid_obj: SolidObject
- step: 执行一次固体时间步
  - dt: 时间步长    

**内部接口**
- _integrate_forces: 计算并应用外力（如重力）
  - dt: 时间步长
- _detect_internal_collisions: 检测自身内部的碰撞(仅软体需要)
  - dt: 时间步长
- _resolve_internal_collisions: 解决自身内部的碰撞(仅软体需要)
  - dt: 时间步长
# Liquid Module
## LiquidObject
表示一个流体实体（粒子集合 + 材质）
- get_particles: 获取当前液体对象的粒子数据（位置、速度等）
  - None
  - particles: list of Particle
- set_particles: 设置当前液体对象的粒子数据
  - particles: list of Particle
- apply_external_forces: 加入重力或者外力.
  - dt, global_forces: dict or None
## LiquidSolver
**外部接口**
- add_liquid: 登记liquid对象
  - liquid_obj: LiquidObject
- step: 执行一次液体时间步
  - dt: 时间步长

**内部接口**
- _P2G: 将例子质量/动量投射到grid
  - None
- _grid_operations: 在grid上执行计算:加重力,更新速度,边界条件,粘性/压力
  - dt: 时间步长
- _G2P: 将grid速度更新回粒子
  - None
# Cloth Module
## ClothObject
表示布料网格（vertices / edges / faces）与其力学属性
- get_vertices: 获取当前布料对象的顶点数据（位置、速度等）
  - None
  - vertices: list of Vertex
- set_vertices: 设置当前布料对象的顶点数据
  - vertices: list of Vertex
- apply_external_forces: 加入重力或者外力.
  - dt, global_forces: dict or None
## ClothSolver
**外部接口**
- add_cloth: 登记cloth对象
  - cloth_obj: ClothObject
- step: 执行一次布料时间步
  - dt: 时间步长

**内部接口**
- _predict_positions: 预测顶点新位置
  - dt: 时间步长
- _solve_internal_constraints: 解决布料内部约束（伸展、弯曲等）
  - dt: 时间步长
- _apply_velocity_updates: 更新顶点速度
  - dt: 时间步长
# Collider
处理固体、流体、布料与环境的碰撞反应
- project_solid_collision: 处理固体对象与碰撞体的碰撞
  - solid_obj: SolidObject, dt: 时间步长
- project_liquid_collision: 处理液体对象与碰撞体的碰撞
  - liquid_obj: LiquidObject, dt: 时间步长
- project_cloth_collision: 处理布料对象与碰撞体的碰撞
  - cloth_obj: ClothObject, dt: 时间步长

Plane / Sphere / Box 等 collider 均继承此类
# Coupling Modules
## SolidLiquidCoupling
实现 solid ↔ liquid 双向作用，包括：
流体施加压力/粘性力到 solid, solid 作为 boundary 影响 fluid
**外部接口**
- step: 执行一次固体-液体耦合计算
  - dt: 时间步长

**内部接口**
- _sample_fluid_on_solid: 在fluid grid 采样速度
  - None
- _apply_fliud_forces_to_solid: 将 fluid 施加到 solid 的力/扭矩计算出来。
  - None
- _apply_solid_boundary_to_fluid: solid 作为 boundary 影响 fluid
  - None
## LiquidClothCoupling
实现 liquid ↔ cloth:
fluid 对 cloth 施加 drag + pressure, cloth 可以吸附流体或影响流动
**外部接口**
- step: 执行一次液体-布料耦合计算
  - dt: 时间步长

**内部接口**
- _compute_drag_forces: 计算流体对布料的阻力
  - None
- _apply_forces_to_cloth: 将阻力施加到布料顶点
  - None
- _modify_fluid_velocity_near_cloth: 修改布料附近的流体速度场, 防止fluid穿cloth
  - None
## SolidClothCoupling
实现 solid ↔ cloth：
检测 cloth 与 solid 的接触, 对 cloth projection, 对 solid 施加反作用力
**外部接口**
- step: 执行一次耦合
  - dt

**内部接口**
- _detect_contacts: 检测 cloth 与 solid 的接触
  - None  
- _solve_cloth_solid_constraints: PBD投影,保持cloth顶点在solid外部
  - None
- _feedback_forces_to_solid: 计算并施加反作用力到 solid
  - None
# Renderer
- draw_solid: 渲染固体对象
  - solid_obj: SolidObject
- draw_liquid: 渲染液体对象
  - liquid_obj: LiquidObject
- draw_cloth: 渲染布料对象
  - cloth_obj: ClothObject

