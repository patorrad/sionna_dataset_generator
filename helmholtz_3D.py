# import numpy as np
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
# import pyvista as pv
# import matplotlib.pyplot as plt

# # Grid setup
# N = 100
# L = 0.5
# h = L / (N - 1)
# n = N ** 3
# wavelength = 0.3
# k = 2 * np.pi / wavelength

# # Indexing helper
# def index(i, j, k):
#     return i + N * (j + N * k)

# # Assemble sparse system
# A = lil_matrix((n, n))
# b = np.zeros(n)

# for i in range(N):
#     for j in range(N):
#         for k_ in range(N):
#             idx = index(i, j, k_)
#             if i in [0, N-1] or j in [0, N-1] or k_ in [0, N-1]:
#                 A[idx, idx] = 1
#                 b[idx] = 0
#             else:
#                 A[idx, idx] = -6 / h**2 - k**2
#                 A[idx, index(i+1, j, k_)] = 1 / h**2
#                 A[idx, index(i-1, j, k_)] = 1 / h**2
#                 A[idx, index(i, j+1, k_)] = 1 / h**2
#                 A[idx, index(i, j-1, k_)] = 1 / h**2
#                 A[idx, index(i, j, k_+1)] = 1 / h**2
#                 A[idx, index(i, j, k_-1)] = 1 / h**2

# # Add point source at center
# center = index(N//2, N//2, N//2)
# b[center] = 1

# # Solve system
# from scipy.sparse import csr_matrix
# u = spsolve(csr_matrix(A), b)
# u_real = np.real(u).reshape((N, N, N))

# # Prepare grid for visualization
# x = y = z = np.linspace(0, L, N)
# grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))
# grid["u"] = u_real.flatten(order="F")

# ### Matplotlib
# # Reshape and visualize a central slice
# u_3d = u.reshape((N, N, N))
# plt.imshow(np.real(u_3d[:, :, N//2]), extent=[0, L, 0, L])
# plt.title("Re(u) Slice at z = L/2")
# plt.colorbar(label="Re(u)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# ### Pyvista
# # # Plot 3D volume
# # plotter = pv.Plotter()
# # plotter.add_volume(grid, scalars="u", opacity="sigmoid", cmap="viridis")
# # plotter.add_axes()
# # plotter.show()

# ### VTK
# # import pyvista as pv
# # import numpy as np

# # # Coordinates
# # x = y = z = np.linspace(0, L, N)

# # # Create a structured grid
# # grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))
# # grid["u_real"] = np.real(u).reshape((N, N, N)).flatten(order="F")

# # # Save to VTK file
# # grid.save("helmholtz_3d.vtk")
# # print("Exported to helmholtz_3d.vtk")

#### Gemini Answer for 2D Helmholtz
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Parameters
# L = 1.0  # Length of the domain
# T = 0.5  # Total time
# c = 1.0  # Wave speed
# nx = 50  # Number of spatial points
# nt = 100 # Number of time points
# dx = L / (nx - 1)
# dt = T / (nt - 1)
# r = c * dt / dx

# # Initialize solution array
# u = np.zeros((nt, nx))

# # Initial conditions
# def initial_condition(x):
#     return np.exp(-(x - 0.5)**2 / 0.01)

# for i in range(nx):
#     u[0, i] = initial_condition(i * dx)
#     if 0 < i < nx -1:
#       u[1,i] = u[0,i] + (r**2/2)*(u[0,i-1] - 2*u[0,i] + u[0,i+1])


# # Time-stepping loop (using forward-time centered-space scheme)
# for n in range(1, nt - 1):
#     for i in range(1, nx - 1):
#         u[n+1, i] = 2*u[n,i] - u[n-1,i] + r**2 * (u[n, i-1] - 2*u[n, i] + u[n, i+1])
#     #Boundary conditions (Dirichlet)
#     u[n + 1, 0] = 0
#     u[n + 1, nx - 1] = 0

# # Visualization
# fig, ax = plt.subplots()
# line, = ax.plot(np.linspace(0, L, nx), u[0, :])
# ax.set_xlabel("x")
# ax.set_ylabel("u(x,t)")
# ax.set_title("Wave Equation Simulation")

# def animate(n):
#     line.set_data(np.linspace(0, L, nx), u[n, :])
#     return line,

# ani = animation.FuncAnimation(fig, animate, frames=nt, interval=50, blit=True)
# plt.show()


### Gemini Answer for 3D Helmholtz
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

def solve_helmholtz_3d(k, f, grid_size, dx):
    """
    Solves the 3D Helmholtz equation using the finite difference method.

    Args:
        k (float): Wave number.
        f (numpy.ndarray): Source term, defined on the grid.
        grid_size (tuple): Number of grid points in each dimension (nx, ny, nz).
        dx (float): Grid spacing.

    Returns:
        numpy.ndarray: Solution u, defined on the grid.
    """
    nx, ny, nz = grid_size
    N = nx * ny * nz

    # 1D Laplacian (tridiagonal)
    def lap1d(n):
        main = -2 * np.ones(n)
        off = np.ones(n - 1)
        return sparse.diags([off, main, off], [-1, 0, 1], shape=(n, n))

    Ix = sparse.identity(nx)
    Iy = sparse.identity(ny)
    Iz = sparse.identity(nz)

    Lx = lap1d(nx)
    Ly = lap1d(ny)
    Lz = lap1d(nz)

    # 3D Laplacian via Kronecker sum: ∇² = Lx⊗Iy⊗Iz + Ix⊗Ly⊗Iz + Ix⊗Iy⊗Lz
    Laplacian = (
        sparse.kron(sparse.kron(Lx, Iy), Iz) +
        sparse.kron(sparse.kron(Ix, Ly), Iz) +
        sparse.kron(sparse.kron(Ix, Iy), Lz)
    ) / dx**2

    # Helmholtz operator: ∇² + k²
    A = Laplacian + k**2 * sparse.identity(N)

    # Reshape source term to vector
    f_vec = f.flatten()

    # Solve the linear system
    u_vec = splinalg.spsolve(A.tocsr(), f_vec)

    # Reshape solution back to grid
    u = u_vec.reshape(grid_size)

    return u

# --- Example usage ---
# k = 2 * np.pi  # wave number
f0 = 2.4e9       # 2.4 GHz WiFi
c = 3e8          # speed of light
k = 2 * np.pi * f0 / c  # wave number

grid_size = (100, 100, 100)
dx = 0.1

# Source: point source in center
f = np.zeros(grid_size)
center = tuple(n // 2 for n in grid_size)
f[center] = 1000

# Solve
u = solve_helmholtz_3d(k, f, grid_size, dx)

import pyvista as pv
import numpy as np

nx, ny, nz = u.shape
x = np.linspace(0, dx * (nx - 1), nx)
y = np.linspace(0, dx * (ny - 1), ny)
z = np.linspace(0, dx * (nz - 1), nz)

# Create structured grid
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
grid = pv.StructuredGrid(X, Y, Z)
grid["u"] = u.flatten(order="F")

grid.save("helmholtz_solution.vtk")