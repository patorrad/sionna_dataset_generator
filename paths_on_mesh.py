import numpy as np

import trimesh
from trimesh.path.entities import Line
from trimesh.path import Path2D, Path3D

from math import tan, sin, cos, sqrt

import matplotlib.pyplot as plt

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

def add_normal_vector(scene, origin, normal, scale=0.5, color=[0, 255, 0, 255]):  
    """
    Adds a normal vector as a line in the scene.

    :param scene: The trimesh scene.
    :param origin: The starting point of the normal vector.
    :param normal: The normal vector direction.
    :param scale: Scaling factor for visualization.
    :param color: Color of the normal vector (default: green).
    """
    # Compute the endpoint of the normal vector
    end_point = origin + normal * scale  # Scale the normal for visibility

    # Create line vertices (start and end)
    line_vertices = np.array([origin, end_point])
    
    # Create line entity
    normal_line = trimesh.load_path(line_vertices, colors=color)

    # Add to scene
    scene.add_geometry(normal_line)


if __name__ == "__main__":

    trajectories = np.empty((0,20,3))
    headings = np.empty((0,20))

    for i in range(1000):
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

        # create some rays
        # ray_origins = track #np.array([[0, 0, -5], [2, 2, -10]])
        # ray_directions = np.array([[0, 0, 1]]*track.shape[0])

        # # run the mesh- ray test
        # locations, index_ray, index_tri = mesh.ray.intersects_location(
        #     ray_origins=ray_origins, ray_directions=ray_directions
        # )          

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
        # ray_visualize = trimesh.load_path(
        #     np.hstack((ray_origins[:1], ray_origins[:1] + ray_directions[:1])).reshape(-1, 2, 3)
        # # 
        # if locations.shape != (20, 3):
        #     import pdb; pdb.set_trace()
        
        # Lidar
        
        # Object to do ray- mesh queries
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=True)
             
        for k in range(len(track)):
            pos = track[k]
            LiDAR_loc = locations[k]
            # add to the z-axis since its on top of the rover
            LiDAR_loc[2] += 0.5
            closest, distance, index = trimesh.proximity.closest_point(mesh, [LiDAR_loc])
            
            nvector = mesh.face_normals[index][0]
            print(nvector)

            # by default is 2D, to make 3D chane to (100, nvector, False) as an example
            ray_directions = spherical_directions.create_LiDAR_direction(100, nvector)          
            ray_origins = [LiDAR_loc] * len(ray_directions)

            index_tri, index_ray, LiDAR_CP = intersector.intersects_id(ray_origins, ray_directions, multiple_hits = False, return_locations = True)
            # print(LiDAR_CP)

            # TESTING
            # check to make sure it works
            scene = trimesh.Scene([mesh])

            # convert all hits on mesh into np array
            LiDAR_CP = np.array(LiDAR_CP, dtype=np.float32)

            # make them all red
            colors = np.full((len(LiDAR_CP), 4), [255, 0, 0, 255], dtype=np.uint8)

            # add the current LiDAR location
            track_point = np.array(LiDAR_loc, dtype=np.float32)  # Ensure it's a 2D array
            hits = np.vstack([LiDAR_CP, track_point])  # Append track[i]

            # add blue color for track point
            new_color = np.array([[0, 0, 255, 255]], dtype=np.uint8)  # Blue point

            # add to scene
            blue_colors = np.vstack([colors, new_color])  # Blue point color
            blue_point_cloud = trimesh.points.PointCloud(hits, colors=blue_colors)
            scene.add_geometry(blue_point_cloud)
            add_normal_vector(scene, track_point, nvector, scale=0.5, color=[[0, 255, 0, 255]])  # Green normal
            
            # scene.show(viewer="gl")
    
        locations = locations[np.argsort(index_ray)]
        locations[:, 2] += 0.5
        locations_reshaped = locations.reshape(1, 20, 3)
        trajectories = np.concatenate((trajectories, locations_reshaped), axis=0) 
        headings = np.concatenate((headings, heading.reshape(1, 20)), axis=0)       

        # scene = trimesh.Scene([mesh, ray_visualize])
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