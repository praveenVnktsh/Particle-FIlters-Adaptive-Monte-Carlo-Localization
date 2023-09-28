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
import matplotlib.pyplot as plt
from params import params, precomputed_cast


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
        true_measurements[idx] = min(params["sensor_model"]["max_range"], precomputed_cast[int(x[0]/10), int(x[1]/10), int(angle)])
        
    return true_measurements
    
  

def raycast_precompute_vec(precomputed_cast, x_t1, z_measured,):
    
    true_measurements = np.zeros((len(x_t1), len(z_measured)))
    theta = x_t1[:, 2]
    
    angles = np.linspace(theta - np.pi/2, theta + np.pi/2, len(z_measured)).T
    angles *= 180/np.pi
    angles = np.mod((angles + 360), 360).astype(int)
    true_measurements = np.zeros((len(x_t1), len(z_measured)))
    
    
    offset = 25
    for i in range(len(z_measured)):
        theta = x_t1[:, 2]
        x = x_t1[:, 0] + offset * np.cos(theta) 
        y = x_t1[:, 1] + offset * np.sin(theta) 
        x = (x/10.0).astype(int)
        y = (y/10.0).astype(int)
        true_measurements[:, i] = precomputed_cast[x, y, angles[:, i]]
    return true_measurements
    


def beam_range_finder_model_vec(z_kt, x_t1):
    
    _sigma_hit= params["sensor_model"]["sigma_hit"]
    _lambda_short = params["sensor_model"]["lambda_short"]
    _z_hit = params["sensor_model"]["z_hit"]
    _z_short = params["sensor_model"]["z_short"]
    _z_max = params["sensor_model"]["z_max"]
    _z_rand = params["sensor_model"]["z_rand"]
    
    
    
    q = 0
    z_kt = z_kt[::params["sensor_model"]["subsampling"]]
    z_kt = np.clip(z_kt, 0, params["sensor_model"]["max_range"])
    z_star_t = raycast_precompute_vec(precomputed_cast, x_t1, z_kt)
    
    p_hit = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    p_max = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    p_short = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    p_rand = np.zeros((z_star_t.shape[0], z_star_t.shape[1]))
    
    p_hit[:, z_kt <= params["sensor_model"]["max_range"]] = np.exp((-1 / 2) * ((z_kt - z_star_t) ** 2) / (_sigma_hit ** 2))
    p_hit[:, z_kt <= params["sensor_model"]["max_range"]] /= _sigma_hit * np.sqrt(2 * np.pi)
    
    p_short = _lambda_short * np.exp(-_lambda_short * z_kt)
    p_short = np.where(z_kt > z_star_t, 0, p_short)
    
    p_max[:, z_kt >= params["sensor_model"]["max_range"]] = 1
    p_rand[:, z_kt < params["sensor_model"]["max_range"]] = (1 / params["sensor_model"]["max_range"])
    
    
    p = _z_hit * p_hit + _z_short * p_short + _z_max * p_max + _z_rand * p_rand
    # print(z_kt.shape, p_hit.shape)
    print(_z_hit * np.sum(p_hit), _z_short * np.sum(p_short), _z_max * np.sum(p_max), _z_rand * np.sum(p_rand))
    q = np.sum(np.log(p), axis = 1)
    q = np.exp(q)
    
    return q, z_star_t


    