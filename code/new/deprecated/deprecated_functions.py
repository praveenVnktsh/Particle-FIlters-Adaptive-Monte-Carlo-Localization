import numpy as np
from numba import njit

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


def beam_range_finder_model(z_t1_arr, x_t1, occupancy_map):
    """
    param[in] z_t1_arr : laser range readings [array of 180 values] at time t
    param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    param[out] prob_zt1 : likelihood of a range scan zt1 at time t
    """
    
    q = 0.
    z_t1_arr = z_t1_arr[::_subsampling]
    z_t1_arr = np.clip(z_t1_arr, 0, params["sensor_model"]["max_range"])
    # z_true_ranges = raycast(occupancy_map, x_t1, z_t1_arr)
    z_true_ranges = raycast_precompute(precomputed_cast, x_t1, z_t1_arr)

    for k in range(len(z_t1_arr)):
        z_measured = z_t1_arr[k]
        z_true = z_true_ranges[k]
        # print(z_measured, z_true)
        if 0 <= z_measured <= params["sensor_model"]["max_range"]:
            p_hit = np.exp((-1 / 2) * ((z_measured - z_true) ** 2) / (_sigma_hit ** 2))
            p_hit /= _sigma_hit * np.sqrt(2 * np.pi)
        else:
            p_hit = 0
            
        if 0 <= z_measured <= z_true: 
            eta = 1
            p_short = eta * _lambda_short * np.exp(-_lambda_short * z_measured)
        else:                
            p_short = 0

        if z_measured >= params["sensor_model"]["max_range"]:
            p_max = params["sensor_model"]["max_range"]
        else:
            p_max = 0
        
        if 0 <= z_measured < params["sensor_model"]["max_range"]:
            p_rand = (1 / params["sensor_model"]["max_range"])
        else:
            p_rand = 0

        p = _z_hit * p_hit + _z_short * p_short + _z_max * p_max + _z_rand * p_rand
        # p /= (_z_hit + _z_short + _z_max + _z_rand)
        # print(p, p_hit, p_short, p_max, p_rand)

        
        q += np.log(p)
    q = np.exp(q)
    return q, z_true_ranges
    
  
@njit
def raycast( occupancy_map, x, zs):
    theta = x[2]

    true_measurements = np.zeros((len(zs), ))
    # for each beam

    for idx, angle in enumerate(np.linspace(theta - np.pi/2, theta + np.pi/2, len(zs))):
        x_t = x[0] #+ laser_frame[0]
        y_t = x[1] #+ laser_frame[1]
        # print(params["sensor_model"]["max_range"], )
        for  dist in (range(0, params["sensor_model"]["max_range"] + resolution, resolution)):
            x_t = x[0] + dist * np.cos(angle)
            y_t = x[1] + dist * np.sin(angle)

            if x_t >= 0 and x_t <= occupancy_map.shape[1] * 10 and y_t >= 0 and y_t <= occupancy_map.shape[0] * 10:
                if occupancy_map[int(y_t/10), int(x_t/10)] > _min_probability or occupancy_map[int(y_t/10), int(x_t/10)] < 0:
                    true_measurements[idx] = (np.sqrt((x_t - x[0]) ** 2 + (y_t - x[1]) ** 2))
                    
                    break
            
    return true_measurements



def visualize_raycast(start_loc, true_measurements, vizmap, col = (0, 0, 255)):
    theta = start_loc[2]
    dist = np.sqrt((laser_frame[0]) ** 2 + (laser_frame[1]) ** 2)
    x_0 = start_loc[0] #+ dist * np.cos(theta)
    y_0 = start_loc[1] #+ dist * np.sin(theta) 
    theta = theta * 180 / np.pi
    # for idx, angle in enumerate(np.linspace(0, 359, 360)):
    for idx, angle in enumerate(np.linspace(theta - 90, theta + 90, 5)):
        angle = np.mod((angle + 360), 360)
        msmt = precomputed_cast[int(x_0/10), int(y_0/10), int(angle)]
        x_t = msmt * np.cos(angle * np.pi / 180) + x_0 
        y_t = msmt * np.sin(angle * np.pi / 180) + y_0
        cv2.line(vizmap, (int(x_0 / 10), int(y_0 / 10)), (int(x_t / 10), int(y_t / 10)), col, 1)
        
    return vizmap