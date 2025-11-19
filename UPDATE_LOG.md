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