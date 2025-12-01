## 12.1
By LKH

- Rewrite API document
- Implement MLS-MPM-CPIC, finish mpm simulation and coupling between mpm objects and rigid bodies
- 

## 11.28
By LHZ

- Implement Solid Simulator version 1, but still facing the problem of falling through the ground.

## 11.27
By LHZ

- Implement Render version 1 for solid boxes collision demo.

## 11.26
By LHZ

- Implement the API document
- In a nutshell, use different simulator classes to manage different material objects, and use coupling method to handle interactions between different materials.

TODOs:
- Implement the precise function
- Decide the specific file directory.

## 11.19

By LKH

- Init repo, implemented MLS-MPM core solver with basic scene authoring functionalities.
- Elastic simulation done, fluid simulation by setting $\mu = 0$.
- 2D preview via Taichi GUI supported.

TODOs:
- More materials: plasticity (should be easy, svd the deformation gradient), cloth (mpm, or coupled with mass-spring?), granular (mpm)
- 3D preview via Taichi GGUI
- Rendering
- Finally, performance optimizations