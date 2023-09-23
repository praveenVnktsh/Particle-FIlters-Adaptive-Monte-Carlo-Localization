'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0
        self._alpha2 = 0.0
        self._alpha3 = 0.0
        self._alpha4 = 0.0


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        
        if np.all(u_t0 == u_t1):
            return x_t0
        
        
        x1, y1, theta1 = u_t0
        x2, y2, theta2 = u_t1
        
        delta_rot_1 = np.arctan2(y2 - y1, x2 - x1) - theta1
        delta_trans = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        delta_rot_2 = theta2 - theta1 - delta_rot_1
        
        
        delta_hat_rot_1 = delta_rot_1 - np.random.normal(0, self._alpha1 * np.power(delta_rot_1, 2) + self._alpha2 * np.power(delta_trans, 2))
        delta_hat_trans = delta_trans - np.random.normal(0, self._alpha3 * np.power(delta_trans, 2) + self._alpha4 * np.power(delta_rot_1, 2) + np.power(delta_rot_2, 2))
        delta_hat_rot_2 = delta_rot_2 - np.random.normal(0, self._alpha1 * np.power(delta_rot_2, 2) + self._alpha2 * np.power(delta_trans, 2))
        
        
        x_t1 = np.array(x_t0) + np.array([
            delta_hat_trans * np.cos(x_t0[2] + delta_hat_rot_1),
            delta_hat_trans * np.sin(x_t0[2] + delta_hat_rot_1),
            delta_hat_rot_1 + delta_hat_rot_2
        ])
        return list(x_t1)
        
