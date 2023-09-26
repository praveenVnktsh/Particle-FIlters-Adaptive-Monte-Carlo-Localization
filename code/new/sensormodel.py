'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import cv2
from numba import njit

_z_hit = 1
_z_short = 0.12
_z_max = 0.05
_z_rand = 100

_sigma_hit = 100
_lambda_short = .1

# Used in p_max and p_rand, optionally in ray casting
_max_range = 8183

# Used for thresholding obstacles of the occupancy map
_min_probability = 0.35
_subsampling = 20
# Used in sampling angles in ray casting
# occupancy_map = occupancy_map
resolution = 5
        # print(occupancy_map.shape)

def set_laser_frame( lf):
    global laser_frame
    laser_frame = lf
    

precomputed_cast = np.load('/home/praveenvnktsh/slam/code/new/precomputed_cast.npy')

@njit
def angleWrap(theta):
    return theta - 2 * np.pi * np.floor((theta+np.pi)/(2*np.pi))

@njit
def raycast_precompute(precomputed_cast, x, zs):
    theta = x[2]
    
    true_measurements = np.zeros((len(zs), ))
    for idx, angle in enumerate(np.linspace(theta - np.pi/2, theta + np.pi/2, len(zs))):
        angle *= 180/np.pi
        angle = np.mod((angle + 360), 360) 
        true_measurements[idx] = precomputed_cast[int(x[0]/10), int(x[1]/10), int(angle)]
        
    return true_measurements
    
    
    
    
@njit
def raycast( occupancy_map, x, zs):
    theta = x[2]

    true_measurements = np.zeros((len(zs), ))
    # for each beam

    for idx, angle in enumerate(np.linspace(theta - np.pi/2, theta + np.pi/2, len(zs))):
        x_t = x[0] #+ laser_frame[0]
        y_t = x[1] #+ laser_frame[1]
        # print(_max_range, )
        for  dist in (range(0, _max_range + resolution, resolution)):
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

def raycast_precompute_vec(precomputed_cast, x_t1, z_measured):
    
    true_measurements = np.zeros((len(x_t1), len(z_measured)))
    theta = x_t1[:, 2]
    
    angles = np.linspace(theta - np.pi/2, theta + np.pi/2, len(z_measured)).T
    angles *= 180/np.pi
    angles = np.mod((angles + 360), 360).astype(int)
    true_measurements = np.zeros((len(x_t1), len(z_measured)))
    for i in range(len(z_measured)):
        true_measurements[:, i] = precomputed_cast[(x_t1[:, 0] / 10).astype(int), (x_t1[:, 1]/10).astype(int), angles[:, i]]
    return true_measurements
    


def beam_range_finder_model_vec(z_kt, x_t1):
    
    q = 0
    z_kt = z_kt[::_subsampling]
    z_kt = np.clip(z_kt, 0, _max_range)
    z_star_t = raycast_precompute_vec(precomputed_cast, x_t1, z_kt)
    
    p_hit = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    p_max = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    p_short = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    p_rand = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    
    p_hit[:, z_kt <= _max_range] = np.exp((-1 / 2) * ((z_kt - z_star_t) ** 2) / (_sigma_hit ** 2))
    p_hit[:, z_kt <= _max_range] /= _sigma_hit * np.sqrt(2 * np.pi)
    
    p_short = _lambda_short * np.exp(-_lambda_short * z_kt)
    p_short = np.where(z_kt > z_star_t, 0, p_short)
    
    p_max[:, z_kt >= _max_range] = _max_range
    p_rand[:, z_kt < _max_range] = (1 / _max_range)
    
    
    p = _z_hit * p_hit + _z_short * p_short + _z_max * p_max + _z_rand * p_rand
    
    q = np.sum(np.log(p), axis = 1)
    q = np.exp(q)
    
    return q, z_star_t

def beam_range_finder_model(z_t1_arr, x_t1, occupancy_map):
    """
    param[in] z_t1_arr : laser range readings [array of 180 values] at time t
    param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    param[out] prob_zt1 : likelihood of a range scan zt1 at time t
    """
    
    q = 0.
    z_t1_arr = z_t1_arr[::_subsampling]
    z_t1_arr = np.clip(z_t1_arr, 0, _max_range)
    # z_true_ranges = raycast(occupancy_map, x_t1, z_t1_arr)
    z_true_ranges = raycast_precompute(precomputed_cast, x_t1, z_t1_arr)

    for k in range(len(z_t1_arr)):
        z_measured = z_t1_arr[k]
        z_true = z_true_ranges[k]
        # print(z_measured, z_true)
        if 0 <= z_measured <= _max_range:
            p_hit = np.exp((-1 / 2) * ((z_measured - z_true) ** 2) / (_sigma_hit ** 2))
            p_hit /= _sigma_hit * np.sqrt(2 * np.pi)
        else:
            p_hit = 0
            
        if 0 <= z_measured <= z_true: 
            eta = 1
            p_short = eta * _lambda_short * np.exp(-_lambda_short * z_measured)
        else:                
            p_short = 0

        if z_measured >= _max_range:
            p_max = _max_range
        else:
            p_max = 0
        
        if 0 <= z_measured < _max_range:
            p_rand = (1 / _max_range)
        else:
            p_rand = 0

        p = _z_hit * p_hit + _z_short * p_short + _z_max * p_max + _z_rand * p_rand
        # p /= (_z_hit + _z_short + _z_max + _z_rand)
        # print(p, p_hit, p_short, p_max, p_rand)
        q += np.log(p)
    q = np.exp(q)
    return q, z_true_ranges
    

    