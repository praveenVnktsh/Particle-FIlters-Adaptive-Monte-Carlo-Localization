from params import params
import numpy as np
import cv2
from numba import njit
from sensormodel import beam_range_finder_model_vec
import imageio


def load_ogm():
    occupancy_map = np.genfromtxt(params["map_path"], skip_header=7)
    occupancy_map[occupancy_map < 0] = -1
    occupancy_map[occupancy_map > 0] = 1 - occupancy_map[occupancy_map > 0]
    occupancy_map = np.flipud(occupancy_map)
    
    return occupancy_map


def init_particles_freespace(num_particles, occupancy_map):
    y, x = np.where(np.logical_and(occupancy_map <= params["min_sampling_probability"], occupancy_map >= 0))
    chosen_indices = np.random.choice(len(x), num_particles)
    x0_vals = x[chosen_indices].reshape(-1, 1) * 10
    y0_vals = y[chosen_indices].reshape(-1, 1) * 10
    
    theta0_vals = np.random.uniform(0, 2*np.pi, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    # print(X_bar_init.shape)
    return X_bar_init

# def kld_sampling(X_bar, timestep):
#     """
#     KLD sampling for adaptive number of particles monte carlo localization algorithm
#     """
#     weights = X_bar[:, 3] / np.sum(X_bar[:, 3])
#     error = np.sum(weights * np.log(weights / np.mean(weights)))
#     if error > 0:
#         print(error)

def resample_particles(X_bar, timestep):
    if timestep > params["sampling_params"]["decay_steps"]: 
        global nparticles
        nparticles = nparticles * np.exp(-params["sampling_params"]["decay_rate"] * (timestep - params["sampling_params"]["decay_steps"]))
        nparticles = max(nparticles, params["sampling_params"]["min_particles"])
        nparticles = int(nparticles)
    
    r1 = np.random.uniform(0, 1/nparticles)

    all_weights = X_bar[:,3]
    all_weights /= np.sum(X_bar[:,3])
    c1 = all_weights[0]
    i=0
    new_particles = []
    
    
    for m in range(0, nparticles):
        u1 = r1 + (m)*(1/nparticles)
        while u1>c1:
            i += 1
            c1 += all_weights[i]
            
        x, y, theta = X_bar[i,:3].flatten()
        
        x += np.random.normal(0, params["sampling_params"]["resampling_noise"]["x"])
        y += np.random.normal(0, params["sampling_params"]["resampling_noise"]["y"])
        theta += np.random.normal(0, params["sampling_params"]["resampling_noise"]["theta"])
        
        new_particles.append(np.array([x, y, theta, 1/nparticles]))

    X_bar_resampled = np.vstack(new_particles)
    return X_bar_resampled


@njit
def angleWrap(theta):
    return np.mod((theta + np.pi), (2 * np.pi)) - np.pi

def motion_model_vec(u_t0, u_t1, x_t0):
    _alpha1 = params["motion_model"]["alpha1"]
    _alpha2 = params["motion_model"]["alpha2"]
    _alpha3 = params["motion_model"]["alpha3"]
    _alpha4 = params["motion_model"]["alpha4"]
    
    
    x1, y1, theta1 = u_t0
    x2, y2, theta2 = u_t1
    
    delta_rot_1 = np.arctan2(y2 - y1, x2 - x1) - theta1
    delta_trans = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    delta_rot_2 = theta2 - theta1 - delta_rot_1
    
    
    
    std1 = _alpha1 * np.power(delta_rot_1, 2) + _alpha2 * np.power(delta_trans, 2)
    std2 = _alpha3 * np.power(delta_trans, 2) + _alpha4 * (np.power(delta_rot_1, 2) + np.power(delta_rot_2, 2))
    std3 = _alpha1 * np.power(delta_rot_2, 2) + _alpha2 * np.power(delta_trans, 2)
    delta_hat_rot_1 = delta_rot_1 - np.random.normal(0, std1, size = x_t0.shape[0])
    delta_hat_trans = delta_trans - np.random.normal(0, std2, size = x_t0.shape[0])
    delta_hat_rot_2 = delta_rot_2 - np.random.normal(0, std3, size = x_t0.shape[0]) 
    x_t1 = np.array(x_t0) + np.array([
        delta_hat_trans * np.cos(x_t0[:, 2] + delta_hat_rot_1),
        delta_hat_trans * np.sin(x_t0[:, 2] + delta_hat_rot_1),
        angleWrap(delta_hat_rot_1 + delta_hat_rot_2)
    ]).T

    return x_t1




nparticles = params["n_particles"]

ogm = load_ogm()
X_bar = init_particles_freespace(nparticles, ogm)
# X_bar = init_particles_random(nparticles, ogm)

logfile = open(params["log_path"], 'r')

viz = ogm.copy()
viz -= viz.min()
viz /= viz.max()

viz *= 255
viz = viz.astype(np.uint8)
ogviz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
def visualize(viz, pause = 1):
    cv2.imshow('sensor_model', viz)
    if cv2.waitKey(pause) == ord('q'):
        exit()
        
        
videoframes = []
import time
starttime = time.time()
for time_idx, line in enumerate(logfile):
    viz = ogviz.copy()
    meas_type = line[0]
    meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
    odom = meas_vals[:3]
    timestamp = meas_vals[-1]
    
    if time_idx == 0 :
        u_t0 = odom
        continue
    
    u_t1 = odom
    if meas_type == "L":
        ranges = meas_vals[6:-1]
        
    
    X_bar_new = np.zeros((nparticles, 4), dtype = np.float64)
    
    x_t1 = motion_model_vec(u_t0, u_t1, X_bar[:, :3])
    
    if meas_type == "L":
        w_t, true_measurements = beam_range_finder_model_vec(ranges, x_t1)
        X_bar_new = np.hstack((x_t1, w_t.reshape(-1, 1)))
    else:
        X_bar_new = np.hstack((x_t1, X_bar[:, 3].reshape(-1, 1)))
    
    X_bar = X_bar_new
    u_t0 = u_t1
    X_bar[:, 3] /= np.sum(X_bar[:, 3])
    if meas_type == "L" and time_idx > params["sampling_params"]["init_steps_without_resampling"]:
        X_bar = resample_particles(X_bar, time_idx)
        
        
        
    # some visualization
    if meas_type == 'L':
        avg_theta = np.average(X_bar_new[:, 2])
        avg_x = np.average(X_bar_new[:, 0]) + 25 * np.cos(avg_theta)
        avg_y = np.average(X_bar_new[:, 1]) + 25 * np.sin(avg_theta)
        for idx, angle in enumerate(np.linspace(avg_theta - np.pi/2, avg_theta + np.pi/2, 180)):
            angle = np.mod((angle + 2 * np.pi), 2*np.pi)
            msmt = min(ranges[idx], params["sensor_model"]["max_range"])
            x_t = msmt * np.cos(angle) + avg_x 
            y_t = msmt * np.sin(angle) + avg_y
            cv2.line(viz, (int(avg_x / 10), int(avg_y / 10)), (int(x_t / 10), int(y_t / 10)), (0, 0, 255), 1)
            

    for m in range(nparticles):
        cv2.circle(viz, (int(x_t1[m, 0]/10), int(x_t1[m, 1]/10)), 3, (0, 255, 0), -1)
    
    if meas_type == "L":
        visualize(viz, 1)
        videoframes.append(viz[:, :, ::-1])
    print(meas_type, "update at timestep", time_idx, "with", nparticles, "particles")    

logfile.close()
runtime = time.time() - starttime
print("Simulation complete in ", runtime , "seconds")
print(len(videoframes))
# logging
import os
import json
timestring = time.strftime("%Y%m%d-%H%M%S")
dirname = f"outputs/{params['log_path'].split('/')[-1].split('.')[0]}_{timestring}/"
os.makedirs(dirname, exist_ok=True)
imageio.mimsave(dirname + 'video.mp4', videoframes, fps = 60)
params["run"] = {
    "simulation_time": runtime,
    "robot_runtime" : timestamp,
    "speed_pct" : (timestamp * 100 / runtime)
}
with open(dirname + 'params.json', 'w') as f:
    # f.write(str(params))
    json.dump(params, f, indent=4)
    
    