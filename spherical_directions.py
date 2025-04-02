import numpy as np
import matplotlib.pyplot as plt

# Finding all ray directions (all directions 3D)
# Convert spherical coordinates to Cartesian coordinates
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

directions = []
for i in range(len(x)):
    for j in range(len(x[i])):
        directions.append([x[i][j], y[i][j], z[i][j]])
        
print(len(directions))

# Testing to make sure that it correctly produces a sphere
import matplotlib.pyplot as plt
import random

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

x = []
y = []
z = []

for dir in directions:
    x.append(dir[0])
    y.append(dir[1])
    z.append(dir[2])

ax.scatter(x, y, z)
# plt.show()