import matplotlib.pyplot as plt
import random
import numpy as np


def create_LiDAR_direction(num, normal_vector):
    u, v = np.mgrid[0:2*np.pi:num*1j, 0:np.pi:num*1j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    
    directions = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            point = np.array([x[i][j], y[i][j], z[i][j]])
            if np.dot(point, normal_vector) >= 0:  # Keep points in the same hemisphere
                directions.append(point)
            # directions.append([x[i][j], y[i][j], z[i][j]])

    return directions


def show_plot(directions):
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
    
    plt.show()

show_plot(create_LiDAR_direction(100, [ 0.05343232, -0.16721065,  0.98447224]))