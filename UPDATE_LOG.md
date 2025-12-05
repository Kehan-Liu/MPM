## 12.5
By LHZ

- Fix the bug after merge
- Solve the unappropriate collision response between rigid bodies and box surfaces.
- Make sure Solid Solid collision works well between balls and ball & bunny

## 12.5
By LKH

- Finish debug CPIC, Jelly-rigid coupling works well now.
- Finish frame rendering for MPM objects.
- Add more demos and meshes.
- Add assets, including background hdri and materials.

TODO: improve liquid simulation quality.

## 12.5
By LKH

- Add surface reconstruction and rendering for MPM objects.
- Rigid-rigid collision has problem, need to be fixed!

## 12.3
By LKH

- Merge rigid body collision handling into the main simulation loop.

## 12.3
By LHZ

- Add trajectory controlled module for rigid body simulator.

## 12.1
By LKH

- Rewrite API document, implement framework
- Implement MLS-MPM-CPIC, finish mpm simulation and coupling between mpm objects and rigid bodies

TODOs:
- merge with rigid body simulator by LHZ
- rendering of mpm objects 

## 12.1
By LHZ

- Fix the bug of solid simulator where solid boxes fall through the ground.
- Fix the bug that the ball bounce too high due to wrong restitution implementation.
- Basically implemented a correct impulse-based rigid body simulator with friction.
- Update the Blender renderer such that it could successfully render the mesh based objects.

Note that the rigid_body_sim_ti.py works as a taichi simulator, use that one!

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