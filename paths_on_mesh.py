import numpy as np

import trimesh
from trimesh.path.entities import Line
from trimesh.path import Path2D, Path3D

from math import tan, sin, cos, sqrt

import matplotlib.pyplot as plt

# import from file that finds all direction vectors
import spherical_directions

dt = 1.0
wheelbase = 0.5
cmds = [np.array([.8, np.random.choice([1, -1]) * .01])] * 20
step = 1

def move(x, dt, u, wheelbase):
    hdg = x[2]
    vel = u[0]
    steering_angle = u[1]
    dist = vel * dt
    if abs(steering_angle) > 0.001: # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle) # radius
        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        return x + np.array([-r*sinh + r*sinhb,
                              r*cosh - r*coshb, beta]), np.array([-vel*sinh + vel*sinhb, vel*cosh - vel*coshb]), hdg
    else: # moving in straight line
        return x + np.array([dist*cos(hdg), dist*sin(hdg), 0]), np.array([vel*cos(hdg), vel*sin(hdg)]), hdg

if __name__ == "__main__":

    trajectories = np.empty((0,20,3))
    headings = np.empty((0,20))

    for i in range(5):
        print(i)
        cmds = [np.array([np.random.choice([1, -1]) * np.random.uniform(.2, .8), np.random.choice([1, -1]) * .01])] * 20
        # Generate trajectories
        track = []
        velocity = []
        heading = []
        sim_pos = np.array([np.random.uniform(10, 50), np.random.uniform(-5, 2), np.random.uniform(0, np.pi / 2)])
        # sim_pos = np.array([30, 5, 0])
        for j, u in enumerate(cmds):
            sim_pos, vel, hdg = move(sim_pos, dt/step, u, wheelbase)
            track_tmp = np.copy(sim_pos)
            track_tmp[2] = -10
            track.append(track_tmp)
            velocity.append(vel)
            heading.append(hdg)
        track = np.array(track)
        velocity = np.array(velocity)
        heading = np.array(heading)

        # plt.plot(track[:, 0], track[:,1], marker='.', color='k', lw=2)
        # plt.axis('equal')
        # plt.title("Robot  Trajectory")
        # plt.show()

        mesh = trimesh.load_mesh("models/canyon.ply")

        # Lidar

        # Object to do ray- mesh queries
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=True)
        
        # Generate rays pointing in all directions
        ray_directions = spherical_directions.directions
        for pos in track:
            ray_origins = [pos] * len(ray_directions)

            LiDAR_CP, index_ray, index_tri = intersector.intersects_location(ray_origins, ray_directions)
            print(LiDAR_CP)

        ray_origins = track #np.array([[0, 0, -5], [2, 2, -10]])
        ray_directions = np.array([[0, 0, 1]]*track.shape[0])

        # run the mesh- ray test
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )
        # import pdb; pdb.set_trace()
        if len(locations) < len(cmds) or locations.shape != (20, 3):
            continue
        # stack rays into line segments for visualization as Path3D
        ray_visualize = trimesh.load_path(
            np.hstack((ray_origins[:1], ray_origins[:1] + ray_directions[:1])).reshape(-1, 2, 3)
        )

        # if locations.shape != (20, 3):
        #     import pdb; pdb.set_trace()
        
        locations = locations[np.argsort(index_ray)]
        locations[:, 2] += 0.5
        locations_reshaped = locations.reshape(1, 20, 3)
        trajectories = np.concatenate((trajectories, locations_reshaped), axis=0) 
        headings = np.concatenate((headings, heading.reshape(1, 20)), axis=0)       

        # ray_origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # ray_directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # ray_visualize = trimesh.load_path(
        #     np.hstack((ray_origins, ray_origins + ray_directions * 5.0)).reshape(-1, 2, 3),
        #     colors=[(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
        # )
        
        # # create a visualization scene with rays, hits, and mesh
        # scene = trimesh.Scene([mesh, ray_visualize, trimesh.points.PointCloud(locations)])

        # Lidar

        # Object to do ray- mesh queries
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=True)
        
        # Generate rays pointing in all directions
        num_rays = 720  # Number of rays per position
        phi = np.linspace(0, np.pi, num_rays//2)  # Elevation angle (0 to π)
        theta = np.linspace(0, 2*np.pi, num_rays//2)  # Azimuth angle (0 to 2π)
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.ravel()
        phi = phi.ravel()

        # Convert spherical coordinates to Cartesian coordinates
        ray_directions = np.vstack([
            np.sin(phi) * np.cos(theta),  # x-component
            np.sin(phi) * np.sin(theta),  # y-component
            np.cos(phi)                   # z-component
        ]).T

        for pos in track:
            ray_origins = [pos] * 720
            
            print(intersector.intersects_location(ray_origins, ray_directions, multiple_hits=True))

        # scene.show()
        # # # mesh.show()
   
    print(f'trajectories {trajectories.shape}')
    import os
    file_path = "trajectories_1000.npy"
    if os.path.exists(file_path):
        existing_data = np.load(file_path)  # Load existing data
        combined_data = np.concatenate((existing_data, trajectories))  # Append
    else:
        combined_data = trajectories  # If file doesn't exist, just save new data

    np.save(file_path, combined_data)
    print(f'headings shape {headings.shape}')
    np.save("headings.npy", heading)