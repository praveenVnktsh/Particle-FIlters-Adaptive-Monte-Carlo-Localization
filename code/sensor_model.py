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
        # self._z_hit = 1
        # self._z_short = 0.1
        # self._z_max = 0.1
        # self._z_rand = 100

        self._z_hit = 150
        self._z_short = 17.5
        self._z_max = 15
        self._z_rand = 100
        self._sigma_hit = 100
        self._lambda_short = 15

        # self._sigma_hit = 50
        # self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183 #/ 10

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.1

        # Used in sampling angles in ray casting
        self._subsampling = 20
        self.occupancy_map = occupancy_map
        self.resolution = 10
        # print(self.occupancy_map.shape)

    def beam_range_finder_model(self, z_t1_arr, x_t1, vizmapog):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        vizmap = vizmapog.copy()
        def raycast(occupancy_map, x, zs):
            theta = x[2] * 180 / np.pi

            true_measurements = np.zeros((len(zs), 1))
            # for each beam

            for idx, angle in enumerate(np.linspace(theta - 90, theta + 90, len(zs))):
                x_t = x[0]
                y_t = x[1]
                # print(self._max_range, )
                for  dist in (range(0, self._max_range + self.resolution, self.resolution)):
                    x_t = x[0] + dist * np.cos(angle * np.pi / 180)
                    y_t = x[1] + dist * np.sin(angle * np.pi / 180)
                    # if x_t <= 0 or x_t >= occupancy_map.shape[1] * 10 or y_t <= 0 or y_t >= occupancy_map.shape[0] * 10:
                    #     break

                    if x_t > 0 and x_t < occupancy_map.shape[1] * 10 and y_t > 0 and y_t < occupancy_map.shape[0] * 10:
                        if occupancy_map[int(y_t/10), int(x_t/10)] > self._min_probability \
                            or occupancy_map[int(y_t/10), int(x_t/10)] < 0:
                            true_measurements[idx] = (np.sqrt((x_t - x[0]) ** 2 + (y_t - x[1]) ** 2))
                            break
                    cv2.circle(vizmap, (int(x_t / 10), int(y_t / 10)), 1, (0, 0, 255), -1)



            return true_measurements, vizmap

        q = 0
        z_t1_arr = z_t1_arr[::self._subsampling]
        z_true_ranges, vizmap = raycast(self.occupancy_map, x_t1, z_t1_arr)


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
            p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
            print(p)
            q = q + np.log(p)
        return q, vizmap