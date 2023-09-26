import numpy as np
import cv2
from numba import njit
from sensormodel import beam_range_finder_model, set_laser_frame, visualize_raycast, beam_range_finder_model_vec, precomputed_cast
import imageio

def load_ogm():
    mappath = '/home/praveenvnktsh/slam/data/map/wean.dat'
    occupancy_map = np.genfromtxt(mappath, skip_header=7)
    occupancy_map[occupancy_map < 0] = -1
    occupancy_map[occupancy_map > 0] = 1 - occupancy_map[occupancy_map > 0]
    occupancy_map = np.flipud(occupancy_map)
    
    return occupancy_map


def init_particles_freespace(num_particles, occupancy_map):
    y, x = np.where(occupancy_map == 0)
    chosen_indices = np.random.choice(len(x), num_particles)
    x0_vals = x[chosen_indices].reshape(-1, 1) * 10
    y0_vals = y[chosen_indices].reshape(-1, 1) * 10
    
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))
    # print(x0_vals)
    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    # print(X_bar_init.shape)
    return X_bar_init


def resample_particles(X_bar):
    num_particles = np.shape(X_bar)[0]
    r1 = np.random.uniform(0, 1/num_particles)
    

    all_weights = X_bar[:,3]
    all_weights /= np.sum(X_bar[:,3])
    c1 = all_weights[0]
    i=0
    new_particles = []
    for m in range(0, num_particles):
        u1 = r1 + (m)*(1/num_particles)
        while u1>c1:
            i += 1
            c1 += all_weights[i]
            
        x, y, theta = X_bar[i,:3].flatten()
        x += np.random.normal(0, 3)
        y += np.random.normal(0, 3)
        theta += np.random.normal(0, 0.05)
        
        new_particles.append(np.array([x, y, theta, 1/num_particles]))

    X_bar_resampled = np.vstack(new_particles)
    return X_bar_resampled


@njit
def angleWrap(theta):
    return np.mod((theta + np.pi), (2 * np.pi)) - np.pi

def motion_model_vec(u_t0, u_t1, x_t0):
    _alpha1 = 0.0005
    _alpha2 = 0.0005
    _alpha3 = 0.006
    _alpha4 = 0.006
    
    
    # if u_t0 == u_t1:
    #     return x_t0
    
    
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

    
    


@njit
def motion_model(u_t0, u_t1, x_t0):
    _alpha1 = 0.0005
    _alpha2 = 0.0005
    _alpha3 = 0.006
    _alpha4 = 0.006
    
    
    if u_t0 == u_t1:
        return x_t0
    
    
    x1, y1, theta1 = u_t0
    x2, y2, theta2 = u_t1
    
    delta_rot_1 = np.arctan2(y2 - y1, x2 - x1) - theta1
    delta_trans = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    delta_rot_2 = theta2 - theta1 - delta_rot_1
    
    
    
    std1 = _alpha1 * np.power(delta_rot_1, 2) + _alpha2 * np.power(delta_trans, 2)
    std2 = _alpha3 * np.power(delta_trans, 2) + _alpha4 * (np.power(delta_rot_1, 2) + np.power(delta_rot_2, 2))
    std3 = _alpha1 * np.power(delta_rot_2, 2) + _alpha2 * np.power(delta_trans, 2)
    delta_hat_rot_1 = delta_rot_1 - np.random.normal(0, std1)
    delta_hat_trans = delta_trans - np.random.normal(0, std2)
    delta_hat_rot_2 = delta_rot_2 - np.random.normal(0, std3) 
    
    x_t1 = np.array(x_t0) + np.array([
        delta_hat_trans * np.cos(x_t0[2] + delta_hat_rot_1),
        delta_hat_trans * np.sin(x_t0[2] + delta_hat_rot_1),
        angleWrap(delta_hat_rot_1 + delta_hat_rot_2)
    ])

    return list(x_t1)

datapath = '/home/praveenvnktsh/slam/data/log/robotdata1.log'
nparticles = 5000

ogm = load_ogm()
X_bar = init_particles_freespace(nparticles, ogm)
# X_bar = init_particles_random(nparticles, ogm)

logfile = open(datapath, 'r')

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
for time_idx, line in enumerate(logfile):
    viz = ogviz.copy()
    meas_type = line[0]
    meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
    odom = meas_vals[:3]
    timestamp = meas_vals[-1]
    
    if time_idx == 0 :
        u_t0 = odom
        continue
    
    if meas_type == "L":
        laser_frame = meas_vals[3:6]
        set_laser_frame(laser_frame)
        ranges = meas_vals[6:-1]
        
    
    X_bar_new = np.zeros((nparticles, 4), dtype = np.float64)
    u_t1 = odom
    
    x_t1 = motion_model_vec(u_t0, u_t1, X_bar[:, :3])
    
    if meas_type == "L":
        w_t, true_measurements = beam_range_finder_model_vec(ranges, x_t1)
        X_bar_new = np.hstack((x_t1, w_t.reshape(-1, 1)))
    else:
        X_bar_new = np.hstack((x_t1, X_bar[:, 3].reshape(-1, 1)))
    
    for m in range(nparticles):
    #     x_t0 = X_bar[m, :3]
    #     x_t1 = motion_model(list(u_t0), list(u_t1), list(x_t0))
        
    #     # X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
    #     if meas_type == "L":
    #         # continue
    #         z_t = ranges
    #         w_t, true_measurements = beam_range_finder_model(z_t, x_t1,  ogm)
    #         # viz = visualize_raycast(x_t1, ranges, viz, (255, 0, 0))    
    #         viz = visualize_raycast(x_t1, true_measurements, viz)    
    #         X_bar_new[m, :] = np.hstack((x_t1, w_t))
    #     else:
    #         X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        cv2.circle(viz, (int(x_t1[m, 0]/10), int(x_t1[m, 1]/10)), 3, (0, 255, 0), -1)
    print(meas_type, time_idx)
    if meas_type == "L":
        visualize(viz, 1)
        videoframes.append(viz[:, :, ::-1])
        
        
    X_bar = X_bar_new
    u_t0 = u_t1
    X_bar[:, 3] /= np.sum(X_bar[:, 3])
    X_bar = resample_particles(X_bar)
imageio.mimsave('sensormodel.mp4', videoframes)
    
    