# import taichi as ti
# import numpy as np
# import matplotlib.pyplot as plt

# ti.init(arch=ti.gpu)  # or ti.cpu

# # Parameters
# nx, ny, nz = 100, 100, 100
# dx = .  # meters
# lam = 0.125  # wavelength for 2.4 GHz
# k_phys = 2 * np.pi / lam  # wavenumber in rad/m
# k = k_phys * dx  # rescale to grid units
# k2 = k**2

# # Taichi fields
# u = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
# u_new = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
# f = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# # Initialize source
# @ti.kernel
# def initialize_source():
#     i = nx // 2
#     j = ny // 2
#     k = nz // 2
#     f[i, j, k] = 1000.0

# # Jacobi iteration kernel
# @ti.kernel
# def helmholtz_iteration():
#     for i, j, k in u:
#         if 0 < i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
#             lap = (
#                 u[i + 1, j, k] + u[i - 1, j, k] +
#                 u[i, j + 1, k] + u[i, j - 1, k] +
#                 u[i, j, k + 1] + u[i, j, k - 1] -
#                 6.0 * u[i, j, k]
#             ) / dx**2
#             u_new[i, j, k] = (f[i, j, k] - lap) / k2

# # Swap u and u_new
# @ti.kernel
# def copy_u():
#     for I in ti.grouped(u):
#         u[I] = u_new[I]

# # Run the solver
# initialize_source()
# for step in range(300):
#     helmholtz_iteration()
#     copy_u()

# # Extract and visualize a cross-section (e.g., mid-z slice)
# u_np = u.to_numpy()
# mid_z = nz // 2

# plt.imshow(np.abs(u_np[:, :, mid_z]), cmap='inferno', origin='lower')
# plt.title('Helmholtz 2D cross-section (z = mid)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar(label='|u|')
# plt.tight_layout()
# plt.savefig('helmholtz_3D_taichi.png', dpi=300)

import taichi as ti
import math
import matplotlib.pyplot as plt
import numpy as np

ti.init(arch=ti.cpu)

# Parameters
N = 32           # grid size in each dimension
L = 1.0         # domain length in meters
dx = L / N       # grid spacing
print(f"dx: {dx} m")
f0 = 2.4e9       # 2.4 GHz WiFi
c = 3e8          # speed of light
k = 2 * np.pi #2 * math.pi * f0 / c  # wave number
alpha = k**2  # Helmholtz equation coefficient

# Fields
u = ti.field(dtype=ti.f32, shape=(N, N, N))       # solution
u_new = ti.field(dtype=ti.f32, shape=(N, N, N))   # for iteration
f = ti.field(dtype=ti.f32, shape=(N, N, N))       # source term

@ti.kernel
def initialize():
    for i, j, k in ti.ndrange(N, N, N):
        u[i, j, k] = 0.0
        u_new[i, j, k] = 0.0
        f[i, j, k] = 0.0

    # Gaussian source at center
    cx, cy, cz = N // 2, N // 2, N // 2
    sigma = 6.0
    for i, j, k in ti.ndrange(N, N, N):
        dx2 = (i - cx)**2 + (j - cy)**2 + (k - cz)**2
        f[i, j, k] = ti.exp(-dx2 / (2 * sigma**2)) * 1000.0

# @ti.kernel
# def jacobi_iteration():
#     for i, j, k in ti.ndrange((1, N-1), (1, N-1), (1, N-1)):
#         laplacian = (
#             u[i-1, j, k] + u[i+1, j, k] +
#             u[i, j-1, k] + u[i, j+1, k] +
#             u[i, j, k-1] + u[i, j, k+1] -
#             6.0 * u[i, j, k]
#         ) / (dx * dx)
#         u_new[i, j, k] = (f[i, j, k] - alpha * u[i, j, k] + laplacian) / (-alpha)

## This method work with N = 1024 and L = 100?
# # Jacobi iteration kernel with relaxation
# @ti.kernel
# def jacobi_iteration():
#     for i, j, k in ti.ndrange((1, N-1), (1, N-1), (1, N-1)):
#         laplacian = (
#             u[i-1, j, k] + u[i+1, j, k] +
#             u[i, j-1, k] + u[i, j+1, k] +
#             u[i, j, k-1] + u[i, j, k+1] -
#             6.0 * u[i, j, k]
#         ) / (dx * dx)

#         rhs = f[i, j, k] + laplacian
#         u_est = rhs / alpha

#         # Relaxation factor (0.5 for safety)
#         omega = 0.5
#         u_new[i, j, k] = omega * u_est + (1 - omega) * u[i, j, k]

@ti.kernel
def jacobi_iteration():
    for i, j, k in ti.ndrange((1, N-1), (1, N-1), (1, N-1)):
        laplacian = (
            u[i-1, j, k] + u[i+1, j, k] +
            u[i, j-1, k] + u[i, j+1, k] +
            u[i, j, k-1] + u[i, j, k+1]
        )
        # u_new[i, j, k] = (f[i, j, k] + laplacian / (dx * dx)) / (6.0 / (dx * dx) + alpha)
        # Note: The factor 6.0 comes from the 6 neighbors in the Laplacian.
        # The division by (dx * dx) normalizes the Laplacian to the grid spacing.
        # The term alpha is already included in the equation, so we divide by it.
        omega = 0.5
        u_est = (f[i, j, k] + laplacian / (dx * dx)) / (6.0 / (dx * dx) + alpha)
        u_new[i, j, k] = omega * u_est + (1.0 - omega) * u[i, j, k]



@ti.kernel
def update():
    for i, j, k in ti.ndrange(N, N, N):
        u[i, j, k] = u_new[i, j, k]

def solve(num_iters=500):
    initialize()
    for iter in range(num_iters):
        jacobi_iteration()
        update()
        if iter % 50 == 0:
            print(f"Iteration {iter}")

solve()

import numpy as np
import matplotlib.pyplot as plt

u_np = u.to_numpy()
print("min:", u_np.min(), "max:", u_np.max())

slice_index = N // 2
plt.imshow(u.to_numpy()[slice_index, :, :], cmap='viridis')
plt.colorbar(label='Field Strength')
plt.title("WiFi signal field (slice at z=mid)")
plt.show()
