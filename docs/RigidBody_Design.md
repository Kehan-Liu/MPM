# 这是一个说明文档,用来解释RigidRigid碰撞的设计
## Step1 计算接触点速度
```python
v_contact_a = a.lin_vel + np.cross(a.ang_vel, c.r_a)
v_contact_b = b.lin_vel + np.cross(b.ang_vel, c.r_b)
v_rel = v_contact_b - v_contact_a
```
用平动速度+自转引起的速度的和来计算接触点的速度
v_rel代表了两个物体的接触点的相对速度
## Step2 取法向速度vn
```python
vn = np.dot(v_rel, c.normal)
```
## Step3 Baumgarte 修正（位置纠正项 bias）
```python
beta = 0.2
bias = -beta * c.penetration / dt
```
用penalty项来修正
## Step 4：计算 "有效质量 (Effective Mass)" denominator
```python
term_a = a.inv_mass + b.inv_mass
term_b = np.dot( n,
                 np.cross(Ia_world(np.cross(c.r_a, n)), c.r_a)
               + np.cross(Ib_world(np.cross(c.r_b, n)), c.r_b)
)
denom = term_a + term_b
```
前半段代表了平动部分的阻力,后半段则代表了转动部分的阻力
## Step 5：求解法向冲量 Jn
```python
j_n = -(1 + e) * vn - bias
j_n = j_n / denom
```
- $-(1+e)*vn$：希望速度反弹到 restitution（反弹系数 $e$）规定的目标速度
- bias：加入穿透修正
- / denom：除以 effective mass 得到实际冲量
## Step 6：施加法向冲量
```python
impulse_n = j_n * n
a.apply_impulse(-impulse_n, c.r_a)
b.apply_impulse( impulse_n, c.r_b)
```
- 冲量方向：沿着法线施加
- A 与 B 方向相反（作用与反作用）
## Step 7：摩擦冲量（切向）
```python
v_t = v_rel - vn * n
t = v_t / ||v_t|| if > 1e-6
j_t = - vt / denom_t
clip |j_t| ≤ μ|j_n|
```
