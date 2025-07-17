import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Grid dimensions
H, W = 100, 2000
k_free = 0.2
k_ground = 1.0

# Define circular hill parameters
diameter = 2500
radius = diameter / 2
x_center = 1000
y_center = H - 1145  # vertical origin is at the bottom

# Create a 1D hill height profile
x_vals = np.arange(W)
hill_profile = np.zeros(W)

for i, x in enumerate(x_vals):
    dx = x - x_center
    if abs(dx) <= radius:
        dy = np.sqrt(radius**2 - dx**2)
        hill_profile[i] = y_center + dy  # peak is at y_center + r
    else:
        hill_profile[i] = H  # below the domain

# Convert to row indices (grid is top-down)
hill_height = H - 1 - np.clip(hill_profile.astype(int), 0, H - 1)

# Wave number map
k_map = np.ones((H, W)) * k_free
for i in range(W):
    k_map[hill_height[i]:, i] = k_ground

# Source position
source_pos = (H - 2, 5)
N = H * W

# Laplacian
diag = -4 * np.ones(N)
off1 = np.ones(N - 1)
off1[np.arange(1, N) % W == 0] = 0
offW = np.ones(N - W)
L = diags([diag, off1, off1, offW, offW], [0, -1, 1, -W, W])

# Solve Helmholtz: (L + k²)ψ = b
# k_sq_flat = (k_map ** 2).flatten()
# A = L + diags(k_sq_flat, 0)
alpha = 0.000005  # absorption coefficient (tune this)
k_complex = k_map + 1j * alpha
k_sq_flat = (k_complex ** 2).flatten()
A = L + diags(k_sq_flat, 0)

b = np.zeros(N)
b[source_pos[0] * W + source_pos[1]] = 1.0

psi_flat = spsolve(A, b)
# psi = psi_flat.reshape((H, W))
psi = psi_flat.reshape((H, W))  # now complex

# ----------------------------------
# 📍 Receiver points (line at ground level)
receiver_y = H - 2
receiver_xs = np.arange(10, W, 10)
receiver_amps = [psi[receiver_y, x] for x in receiver_xs]
receiver_dists = np.sqrt((receiver_y - source_pos[0])**2 + (receiver_xs - source_pos[1])**2)

# Convert amplitude to dB path loss (arbitrary scale)
path_loss_db = -20 * np.log10(np.abs(receiver_amps) + 1e-8)

# ----------------------------------
# 📊 Plot Signal Field
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(psi), cmap='viridis', origin='lower')
plt.colorbar(label='Signal Strength (ψ)')
plt.plot(range(W), hill_height, color='black', label='Terrain')
plt.scatter(source_pos[1], source_pos[0], color='red', label='Source')
plt.scatter(receiver_xs, [receiver_y]*len(receiver_xs), color='white', s=10, label='Receivers')
plt.title('Signal Field Over Terrain (Helmholtz)')
plt.xlabel('x'); plt.ylabel('y')
plt.legend()

# ----------------------------------
# 📈 Plot Path Loss vs Distance
plt.subplot(1, 2, 2)
plt.plot(receiver_dists, path_loss_db, marker='o')
plt.title('Estimated Path Loss vs Distance')
plt.xlabel('Distance from Source (grid units)')
plt.ylabel('Path Loss (dB)')
plt.grid(True)

plt.tight_layout()
plt.savefig("helmholtz_hill.png", dpi=300)
