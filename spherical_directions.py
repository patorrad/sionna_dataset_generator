import matplotlib.pyplot as plt
import random
import numpy as np


def create_LiDAR_direction(num, normal_vector, planar=True):
    if (planar):
        return create_LiDAR_direction_2D(num, normal_vector)
    
    u, v = np.mgrid[0:2*np.pi:num*1j, 0:np.pi:num*1j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    
    directions = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            point = np.array([x[i][j], y[i][j], z[i][j]])
            dot = np.dot(point, normal_vector)

            if dot >= 0:  # Keep points in the same hemisphere
                directions.append(point)


    return directions

def create_LiDAR_direction_2D(num, normal_vector):
    # For a complete circle perpendicular to the normal vector:
    
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Create basis vectors for the plane perpendicular to normal_vector
    if abs(normal_vector[0]) < abs(normal_vector[1]) and abs(normal_vector[0]) < abs(normal_vector[2]):
        v1 = np.array([0, -normal_vector[2], normal_vector[1]])
    elif abs(normal_vector[1]) < abs(normal_vector[2]):
        v1 = np.array([-normal_vector[2], 0, normal_vector[0]])
    else:
        v1 = np.array([-normal_vector[1], normal_vector[0], 0])
        
    v1 = v1 / np.linalg.norm(v1)  # Normalize v1
    v2 = np.cross(normal_vector, v1)  # v2 is perpendicular to both normal_vector and v1
    
    # Use linspace with endpoint=True to ensure inclusion of both endpoints
    theta = np.linspace(0, 2*np.pi, num, endpoint=True)
    directions = []
    
    for t in theta:
        point = np.cos(t) * v1 + np.sin(t) * v2
        directions.append(point)
    
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

show_plot(create_LiDAR_direction(100, [ 0, 0.5,  1]))
show_plot(create_LiDAR_direction(100, [ 0, 0.5,  1], False))