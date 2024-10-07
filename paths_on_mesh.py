import numpy as np

import trimesh

from math import tan, sin, cos, sqrt

import matplotlib.pyplot as plt

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

    for i in range(20):
        cmds = [np.array([np.random.choice([1, -1]) * np.random.uniform(.2, .8), np.random.choice([1, -1]) * .01])] * 20
        # Generate trajectories
        track = []
        velocity = []
        heading = []
        sim_pos = np.array([np.random.uniform(10, 50), np.random.uniform(-5, 2), np.random.uniform(0, np.pi / 2)])
        # sim_pos = np.array([30, 5, 0])
        for i, u in enumerate(cmds):
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
        
        print(trajectories.shape)
        
        # create a visualization scene with rays, hits, and mesh
        scene = trimesh.Scene([mesh, ray_visualize, trimesh.points.PointCloud(locations)])

        scene.show()
        # # mesh.show()
   
    np.save("trajectories.npy", trajectories)