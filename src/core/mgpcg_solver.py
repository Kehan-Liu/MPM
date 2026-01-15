import taichi as ti
import numpy as np


@ti.data_oriented
class MGPCGSolver:
    def __init__(
        self,
        n_grid: int,
        dx: float,
        dt: float,
        n_mg_levels: int = 4,
        pre_smoothing_iters: int = 2,
        post_smoothing_iters: int = 2,
        bottom_smoothing_iters: int = 50,
        cg_max_iters: int = 500,
        cg_tolerance: float = 1e-6,
        omega: float = 0.67,
    ):
        self.n_grid = n_grid
        self.dx = dx
        self.dt = dt
        self.inv_dx = 1.0 / dx

        # Multigrid
        self.n_mg_levels = n_mg_levels
        self.pre_smoothing_iters = pre_smoothing_iters
        self.post_smoothing_iters = post_smoothing_iters
        self.bottom_smoothing_iters = bottom_smoothing_iters
        self.omega = omega

        # CG
        self.cg_max_iters = cg_max_iters
        self.cg_tolerance = cg_tolerance

        self.grid_sizes = []
        size = n_grid
        for _ in range(n_mg_levels):
            self.grid_sizes.append(size)
            size = max(size // 2, 4)

        print(f"[MGPCG] Initializing with {n_mg_levels} levels: {self.grid_sizes}")

        self.pressure = []
        self.rhs = []
        self.residual = []
        self.temp = []

        for level in range(n_mg_levels):
            size = self.grid_sizes[level]
            shape = (size, size, size)
            self.pressure.append(ti.field(dtype=ti.f32, shape=shape))
            self.rhs.append(ti.field(dtype=ti.f32, shape=shape))
            self.residual.append(ti.field(dtype=ti.f32, shape=shape))
            self.temp.append(ti.field(dtype=ti.f32, shape=shape))

        # Cell type: 0 = air (Dirichlet p=0), 1 = fluid, 2 = solid (Neumann)
        self.cell_type = ti.field(dtype=ti.i32, shape=(n_grid, n_grid, n_grid))

        finest_shape = (n_grid, n_grid, n_grid)
        self.cg_r = ti.field(dtype=ti.f32, shape=finest_shape)
        self.cg_z = ti.field(dtype=ti.f32, shape=finest_shape)
        self.cg_p = ti.field(dtype=ti.f32, shape=finest_shape)
        self.cg_Ap = ti.field(dtype=ti.f32, shape=finest_shape)

        self.cg_alpha = ti.field(dtype=ti.f32, shape=())
        self.cg_beta = ti.field(dtype=ti.f32, shape=())
        self.cg_rz = ti.field(dtype=ti.f32, shape=())
        self.cg_rz_new = ti.field(dtype=ti.f32, shape=())
        self.cg_pAp = ti.field(dtype=ti.f32, shape=())
        self.cg_residual_norm = ti.field(dtype=ti.f32, shape=())
        self.cg_rhs_norm = ti.field(dtype=ti.f32, shape=())

        self.CELL_AIR = 0
        self.CELL_FLUID = 1
        self.CELL_SOLID = 2

    @ti.func
    def is_valid(self, I, size: int) -> bool:
        return 0 <= I[0] < size and 0 <= I[1] < size and 0 <= I[2] < size

    @ti.func
    def neighbor_coeff(self, I, neighbor, level: int) -> ti.f32:
        size = self.grid_sizes[level]
        coeff = 0.0
        if self.is_valid(neighbor, size):
            if level == 0:
                # Finest level: use actual boundary conditions
                if self.cell_type[neighbor] == self.CELL_FLUID:
                    coeff = 1.0
                elif self.cell_type[neighbor] == self.CELL_AIR:
                    coeff = 1.0  # Dirichlet: contributes to diagonal only
                # CELL_SOLID: Neumann, no contribution
            else:
                # Coarser levels: simple Laplacian
                coeff = 1.0
        return coeff

    @ti.kernel
    def compute_divergence(self, grid_v: ti.template()):
        for I in ti.grouped(self.rhs[0]):
            if self.cell_type[I] == self.CELL_FLUID:
                div = 0.0
                # Central difference for divergence
                for d in ti.static(range(3)):
                    offset = ti.Vector.zero(ti.i32, 3)
                    offset[d] = 1
                    I_p = I + offset
                    I_m = I - offset
                    v_p = 0.0
                    v_m = 0.0
                    if self.is_valid(I_p, self.n_grid):
                        v_p = grid_v[I_p][d]
                    if self.is_valid(I_m, self.n_grid):
                        v_m = grid_v[I_m][d]
                    div += (v_p - v_m) * 0.5 * self.inv_dx
                self.rhs[0][I] = div / self.dt
            else:
                self.rhs[0][I] = 0.0

    @ti.kernel
    def jacobi_smooth_kernel(self, level: int):
        size = self.grid_sizes[level]
        omega = self.omega
        h2 = (self.dx * (1 << level)) ** 2

        for I in ti.grouped(self.temp[level]):
            if I[0] >= size or I[1] >= size or I[2] >= size:
                continue
            if level == 0 and self.cell_type[I] != self.CELL_FLUID:
                self.temp[level][I] = 0.0
                continue

            neighbor_sum = 0.0
            diag = 0.0

            for d in ti.static(range(3)):
                for s in ti.static([-1, 1]):
                    offset = ti.Vector.zero(ti.i32, 3)
                    offset[d] = s
                    neighbor = I + offset
                    c = self.neighbor_coeff(I, neighbor, level)
                    if c > 0 and self.is_valid(neighbor, size):
                        if level == 0 and self.cell_type[neighbor] == self.CELL_FLUID:
                            neighbor_sum += self.pressure[level][neighbor]
                        elif level > 0:
                            neighbor_sum += self.pressure[level][neighbor]
                    diag += c

            if diag > 0:
                x_jacobi = (neighbor_sum - h2 * self.rhs[level][I]) / diag
                self.temp[level][I] = (
                    omega * x_jacobi + (1.0 - omega) * self.pressure[level][I]
                )
            else:
                self.temp[level][I] = self.pressure[level][I]

    @ti.kernel
    def copy_temp_to_pressure(self, level: int):
        size = self.grid_sizes[level]
        for I in ti.grouped(self.pressure[level]):
            if I[0] < size and I[1] < size and I[2] < size:
                self.pressure[level][I] = self.temp[level][I]

    def smooth(self, level: int, n_iters: int):
        for _ in range(n_iters):
            self.jacobi_smooth_kernel(level)
            self.copy_temp_to_pressure(level)

    @ti.kernel
    def compute_residual_kernel(self, level: int):
        size = self.grid_sizes[level]
        h2 = (self.dx * (1 << level)) ** 2

        for I in ti.grouped(self.residual[level]):
            if I[0] >= size or I[1] >= size or I[2] >= size:
                continue
            if level == 0 and self.cell_type[I] != self.CELL_FLUID:
                self.residual[level][I] = 0.0
                continue

            center = self.pressure[level][I]
            neighbor_sum = 0.0
            diag = 0.0

            for d in ti.static(range(3)):
                for s in ti.static([-1, 1]):
                    offset = ti.Vector.zero(ti.i32, 3)
                    offset[d] = s
                    neighbor = I + offset
                    c = self.neighbor_coeff(I, neighbor, level)
                    if c > 0 and self.is_valid(neighbor, size):
                        if level == 0 and self.cell_type[neighbor] == self.CELL_FLUID:
                            neighbor_sum += self.pressure[level][neighbor]
                        elif level > 0:
                            neighbor_sum += self.pressure[level][neighbor]
                    diag += c

            Ax = -(neighbor_sum - diag * center) / h2
            self.residual[level][I] = self.rhs[level][I] - Ax

    @ti.kernel
    def restrict_kernel(self, fine_level: int):
        coarse_level = fine_level + 1
        fine_size = self.grid_sizes[fine_level]
        coarse_size = self.grid_sizes[coarse_level]

        for I in ti.grouped(self.rhs[coarse_level]):
            if I[0] >= coarse_size or I[1] >= coarse_size or I[2] >= coarse_size:
                continue

            fine_base = I * 2
            total = 0.0
            count = 0.0

            for di in ti.static(range(2)):
                for dj in ti.static(range(2)):
                    for dk in ti.static(range(2)):
                        fine_idx = fine_base + ti.Vector([di, dj, dk])
                        if self.is_valid(fine_idx, fine_size):
                            total += self.residual[fine_level][fine_idx]
                            count += 1.0

            self.rhs[coarse_level][I] = total / ti.max(count, 1.0)

    @ti.kernel
    def prolongate_kernel(self, fine_level: int):
        coarse_level = fine_level + 1
        fine_size = self.grid_sizes[fine_level]
        coarse_size = self.grid_sizes[coarse_level]

        for I in ti.grouped(self.pressure[fine_level]):
            if I[0] >= fine_size or I[1] >= fine_size or I[2] >= fine_size:
                continue

            coarse_pos = (I.cast(ti.f32) + 0.5) * 0.5 - 0.5
            base = ti.cast(ti.floor(coarse_pos), ti.i32)
            frac = coarse_pos - base.cast(ti.f32)

            result = 0.0
            for di in ti.static(range(2)):
                for dj in ti.static(range(2)):
                    for dk in ti.static(range(2)):
                        c_idx = base + ti.Vector([di, dj, dk])
                        c_idx = ti.max(0, ti.min(c_idx, coarse_size - 1))

                        w = 1.0
                        w *= (1.0 - frac[0]) if di == 0 else frac[0]
                        w *= (1.0 - frac[1]) if dj == 0 else frac[1]
                        w *= (1.0 - frac[2]) if dk == 0 else frac[2]

                        result += w * self.pressure[coarse_level][c_idx]

            self.pressure[fine_level][I] += result

    @ti.kernel
    def zero_pressure_kernel(self, level: int):
        size = self.grid_sizes[level]
        for I in ti.grouped(self.pressure[level]):
            if I[0] < size and I[1] < size and I[2] < size:
                self.pressure[level][I] = 0.0

    def v_cycle(self, level: int = 0):
        if level == self.n_mg_levels - 1:
            self.smooth(level, self.bottom_smoothing_iters)
            return

        self.smooth(level, self.pre_smoothing_iters)
        self.compute_residual_kernel(level)
        self.restrict_kernel(level)
        self.zero_pressure_kernel(level + 1)
        self.v_cycle(level + 1)
        self.prolongate_kernel(level)
        self.smooth(level, self.post_smoothing_iters)

    @ti.kernel
    def cg_init(self):
        h2 = self.dx * self.dx
        self.cg_rhs_norm[None] = 0.0
        self.cg_residual_norm[None] = 0.0

        for I in ti.grouped(self.cg_r):
            if self.cell_type[I] != self.CELL_FLUID:
                self.cg_r[I] = 0.0
                continue

            center = self.pressure[0][I]
            neighbor_sum = 0.0
            diag = 0.0

            for d in ti.static(range(3)):
                for s in ti.static([-1, 1]):
                    offset = ti.Vector.zero(ti.i32, 3)
                    offset[d] = s
                    neighbor = I + offset
                    if self.is_valid(neighbor, self.n_grid):
                        if self.cell_type[neighbor] == self.CELL_FLUID:
                            neighbor_sum += self.pressure[0][neighbor]
                            diag += 1.0
                        elif self.cell_type[neighbor] == self.CELL_AIR:
                            diag += 1.0

            Ax = -(neighbor_sum - diag * center) / h2
            self.cg_r[I] = self.rhs[0][I] - Ax

            ti.atomic_add(self.cg_rhs_norm[None], self.rhs[0][I] ** 2)
            ti.atomic_add(self.cg_residual_norm[None], self.cg_r[I] ** 2)

    @ti.kernel
    def cg_compute_Ap_and_pAp(self):
        h2 = self.dx * self.dx
        self.cg_pAp[None] = 0.0

        for I in ti.grouped(self.cg_Ap):
            if self.cell_type[I] != self.CELL_FLUID:
                self.cg_Ap[I] = 0.0
                continue

            center = self.cg_p[I]
            neighbor_sum = 0.0
            diag = 0.0

            for d in ti.static(range(3)):
                for s in ti.static([-1, 1]):
                    offset = ti.Vector.zero(ti.i32, 3)
                    offset[d] = s
                    neighbor = I + offset
                    if self.is_valid(neighbor, self.n_grid):
                        if self.cell_type[neighbor] == self.CELL_FLUID:
                            neighbor_sum += self.cg_p[neighbor]
                            diag += 1.0
                        elif self.cell_type[neighbor] == self.CELL_AIR:
                            diag += 1.0

            self.cg_Ap[I] = -(neighbor_sum - diag * center) / h2
            ti.atomic_add(self.cg_pAp[None], self.cg_p[I] * self.cg_Ap[I])

    @ti.kernel
    def cg_update_x_r(self):
        alpha = self.cg_alpha[None]
        self.cg_residual_norm[None] = 0.0
        for I in ti.grouped(self.pressure[0]):
            if self.cell_type[I] == self.CELL_FLUID:
                self.pressure[0][I] += alpha * self.cg_p[I]
                self.cg_r[I] -= alpha * self.cg_Ap[I]
                ti.atomic_add(self.cg_residual_norm[None], self.cg_r[I] ** 2)

    @ti.kernel
    def cg_compute_rz(self):
        self.cg_rz[None] = 0.0
        for I in ti.grouped(self.cg_r):
            if self.cell_type[I] == self.CELL_FLUID:
                ti.atomic_add(self.cg_rz[None], self.cg_r[I] * self.cg_z[I])

    @ti.kernel
    def cg_update_p(self, first_iter: int):
        beta = self.cg_beta[None]
        for I in ti.grouped(self.cg_p):
            if self.cell_type[I] == self.CELL_FLUID:
                if first_iter:
                    self.cg_p[I] = self.cg_z[I]
                else:
                    self.cg_p[I] = self.cg_z[I] + beta * self.cg_p[I]
            else:
                self.cg_p[I] = 0.0

    @ti.kernel
    def copy_r_to_rhs(self):
        for I in ti.grouped(self.rhs[0]):
            self.rhs[0][I] = self.cg_r[I]

    @ti.kernel
    def copy_pressure_to_z(self):
        for I in ti.grouped(self.cg_z):
            self.cg_z[I] = self.pressure[0][I]

    def apply_preconditioner(self):
        self.copy_r_to_rhs()
        self.zero_pressure_kernel(0)
        self.v_cycle(0)
        self.copy_pressure_to_z()

    def solve_cg(self) -> int:
        self.cg_init()

        rhs_norm = np.sqrt(self.cg_rhs_norm[None])
        if rhs_norm < 1e-12:
            return 0

        self.apply_preconditioner()
        self.cg_compute_rz()
        rz_old = self.cg_rz[None]
        self.cg_update_p(1)  # p = z

        for iteration in range(self.cg_max_iters):
            self.cg_compute_Ap_and_pAp()

            pAp = self.cg_pAp[None]
            if abs(pAp) < 1e-15:
                break

            self.cg_alpha[None] = rz_old / pAp
            self.cg_update_x_r()

            residual_norm = np.sqrt(self.cg_residual_norm[None])
            if residual_norm / rhs_norm < self.cg_tolerance:
                if iteration % 50 == 0 or iteration < 5:
                    print(
                        f"[MGPCG] Converged in {iteration + 1} iters, residual = {residual_norm / rhs_norm:.2e}"
                    )
                return iteration + 1

            self.apply_preconditioner()
            self.cg_compute_rz()
            rz_new = self.cg_rz[None]

            if abs(rz_old) < 1e-15:
                break

            self.cg_beta[None] = rz_new / rz_old
            rz_old = rz_new
            self.cg_update_p(0)  # p = z + beta * p

        print(f"[MGPCG] Did not converge in {self.cg_max_iters} iterations")
        return self.cg_max_iters

    @ti.kernel
    def apply_pressure_gradient(self, grid_v: ti.template()):
        scale = self.dt * self.inv_dx

        for I in ti.grouped(grid_v):
            if self.cell_type[I] == self.CELL_FLUID:
                grad_p = ti.Vector.zero(ti.f32, 3)
                p_center = self.pressure[0][I]

                for d in ti.static(range(3)):
                    offset = ti.Vector.zero(ti.i32, 3)
                    offset[d] = 1

                    I_p = I + offset
                    I_m = I - offset

                    p_p = p_center  # Default: Neumann (use center value)
                    p_m = p_center

                    if self.is_valid(I_p, self.n_grid):
                        if self.cell_type[I_p] == self.CELL_FLUID:
                            p_p = self.pressure[0][I_p]
                        elif self.cell_type[I_p] == self.CELL_AIR:
                            p_p = 0.0  # Dirichlet

                    if self.is_valid(I_m, self.n_grid):
                        if self.cell_type[I_m] == self.CELL_FLUID:
                            p_m = self.pressure[0][I_m]
                        elif self.cell_type[I_m] == self.CELL_AIR:
                            p_m = 0.0

                    grad_p[d] = (p_p - p_m) * 0.5

                grid_v[I] -= scale * grad_p

    @ti.kernel
    def classify_cells(self, grid_m: ti.template(), mass_threshold: float):
        for I in ti.grouped(self.cell_type):
            if grid_m[I] > mass_threshold:
                self.cell_type[I] = self.CELL_FLUID
            elif (
                I[0] <= 2
                or I[0] >= self.n_grid - 3
                or I[1] <= 2
                or I[1] >= self.n_grid - 3
                or I[2] <= 2
                or I[2] >= self.n_grid - 3
            ):
                self.cell_type[I] = self.CELL_SOLID
            else:
                self.cell_type[I] = self.CELL_AIR

    @ti.kernel
    def classify_cells_with_rigid(
        self,
        grid_m: ti.template(),
        grid_A: ti.template(),
        mass_threshold: float,
        n_rigid: int,
    ):
        for I in ti.grouped(self.cell_type):
            is_solid = False
            for r in range(n_rigid):
                if grid_A[I][r] == 1:
                    is_solid = True
                    break

            if is_solid:
                self.cell_type[I] = self.CELL_SOLID
            elif grid_m[I] > mass_threshold:
                self.cell_type[I] = self.CELL_FLUID
            elif (
                I[0] <= 2
                or I[0] >= self.n_grid - 3
                or I[1] <= 2
                or I[1] >= self.n_grid - 3
                or I[2] <= 2
                or I[2] >= self.n_grid - 3
            ):
                self.cell_type[I] = self.CELL_SOLID
            else:
                self.cell_type[I] = self.CELL_AIR

    def project(
        self,
        grid_v,
        grid_m,
        mass_threshold: float = 1e-6,
        grid_A=None,
        n_rigid: int = 0,
    ) -> int:
        # 1. Classify cells
        if grid_A is not None and n_rigid > 0:
            self.classify_cells_with_rigid(grid_m, grid_A, mass_threshold, n_rigid)
        else:
            self.classify_cells(grid_m, mass_threshold)

        # 2. Compute divergence (RHS)
        self.compute_divergence(grid_v)

        # 3. Solve pressure Poisson equation
        self.zero_pressure_kernel(0)
        n_iters = self.solve_cg()

        # 4. Apply pressure gradient
        self.apply_pressure_gradient(grid_v)

        return n_iters
