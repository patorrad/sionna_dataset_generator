import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Grid size
H, W = 100, 100
dx = dy = 0.01  # meters
k_phys = 2 * np.pi / 0.125
k_grid = k_phys * dx  # ~0.5027
k_free = np.ones((H, W)) * k_grid
# k_free =  2 * np.pi / 0.125       # wave number in free space
k_building = 1.0    # wave number in buildings

# Create domain
psi = np.zeros((H, W))
k_map = np.ones((H, W)) * k_free
building_mask = np.zeros((H, W), dtype=bool)

# Place buildings (vertical wall)
building_mask[:, 45:55] = True
k_map[building_mask] = k_building

# Place source
source_pos = (50, 25)
psi[source_pos] = 1.0  # Dirichlet source condition

# Laplacian operator in 2D using 5-point stencil (flattened)
N = H * W
diag = -4 * np.ones(N)
off1 = np.ones(N - 1)
off1[np.arange(1, N) % W == 0] = 0  # prevent wraparound
offW = np.ones(N - W)
L = diags([diag, off1, off1, offW, offW], [0, -1, 1, -W, W])

# Flatten k^2 * psi
k_sq_flat = (k_map ** 2).flatten()
A = L + diags(k_sq_flat, 0)
# alpha = 0.000005  # absorption coefficient (tune this)
# k_complex = k_map + 1j * alpha
# k_sq_flat = (k_complex ** 2).flatten()
# A = L + diags(k_sq_flat, 0)

# Build b vector (Dirichlet source)
b = np.zeros(N)
b[source_pos[0] * W + source_pos[1]] = 1.0  # unit source

# Solve: (L + k^2) * psi = b
psi_flat = spsolve(A, b)
# psi = psi_flat.reshape((H, W))
psi = psi_flat.reshape((H, W))  # now complex

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot the signal field
im0 = axs[0].imshow(np.abs(psi), cmap='viridis', origin='lower')
axs[0].set_title('Signal Strength (ψ)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].scatter(source_pos[1], source_pos[0], color='red', label='Source')
axs[0].legend()
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# Plot the building mask
im1 = axs[1].imshow(building_mask, cmap='gray', origin='lower')
axs[1].set_title('Building Mask')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('hemlholtz_2D.png', dpi=300)
