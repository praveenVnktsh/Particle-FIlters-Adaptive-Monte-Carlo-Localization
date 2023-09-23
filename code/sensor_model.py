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
from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000 #/ 10

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 1
        self.occupancy_map = occupancy_map
        self.resolution = 1
        # print(self.occupancy_map.shape)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        def raycast(occupancy_map, x, zs):
            theta = x[2] * 180 / np.pi
            vizmap = cv2.resize(occupancy_map, (800, 800), interpolation=cv2.INTER_NEAREST)
            vizmap -= np.min(vizmap)
            vizmap /= np.max(vizmap)
            vizmap *= 255
            vizmap = vizmap.astype(np.uint8)
            vizmap = cv2.cvtColor(vizmap, cv2.COLOR_GRAY2BGR)
            true_measurements = []
            # for each beam
            for angle in np.linspace(theta - 90, theta + 90, len(zs)):
                x_t = x[0]
                y_t = x[1]
                
                for j in range(self._max_range):
                    x_t += self.resolution * np.cos(angle * np.pi / 180)
                    y_t += self.resolution * np.sin(angle * np.pi / 180)
                    if x_t < 0 or x_t >= occupancy_map.shape[0] or y_t < 0 or y_t >= occupancy_map.shape[1]:
                        true_measurements.append(self._max_range)
                        break
                    if occupancy_map[int(x_t), int(y_t)] > self._min_probability:
                        # return np.sqrt((x_t - x[0]) ** 2 + (y_t - x[1]) ** 2)
                        true_measurements.append(np.sqrt((x_t - x[0]) ** 2 + (y_t - x[1]) ** 2))
                        break
                    
                    cv2.circle(vizmap, (int(x_t / 10), int(y_t / 10)), 3, (0, 0, 255), -1)
                    
            return true_measurements, vizmap

        q = 1
        z_t1_arr = z_t1_arr[::self._subsampling]
        z_true_ranges, vizmap = raycast(self.occupancy_map, x_t1, z_t1_arr)
        cv2.imshow('maap', vizmap)
        cv2.waitKey(0)
        
        # print(np.linalg.norm(z_t1_arr - z_true_ranges))        
        
        for k in range(len(z_t1_arr)):
            z_measured = z_t1_arr[k]
            z_true = z_true_ranges[k]
            p_hit = norm.pdf(z_measured, loc = z_true, scale = self._sigma_hit)
            if z_measured > self._z_max:
                p_hit = 0
            
            p_short = self._lambda_short * np.exp(-self._lambda_short * z_measured)
            if z_measured > z_true:
                p_short = 0
                
            
            p_max = (z_measured == self._max_range)
            p_rand = (1 / self._max_range)
            if z_measured > self._max_range:
                p_rand = 0
            
            p = self._z_hit * p_hit + self._z_short * p_short + self._z_max * p_max + self._z_rand * p_rand
            q = q * p
        return q