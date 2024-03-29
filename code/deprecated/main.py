'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

import time

import cv2


def viz_map_particles(occupancy_map, x):
    resized_map = cv2.resize(occupancy_map, (800, 800), interpolation=cv2.INTER_NEAREST)
    resized_map -= np.min(resized_map)
    resized_map /= np.max(resized_map)
    resized_map *= 255
    resized_map = resized_map.astype(np.uint8)
    for xx in x:
        cv2.circle(resized_map, (int(xx[0]/10 ), int(xx[1]/10 )), 2, (0, 0, 255), -1)
    cv2.imshow('map', resized_map)
    if cv2.waitKey(1) == ord('q'):
        exit(0)

def visualize_map(occupancy_map):
    print(occupancy_map.shape)
    resized_map = cv2.resize(occupancy_map, (800, 800), interpolation=cv2.INTER_NEAREST)
    resized_map -= np.min(resized_map)
    resized_map /= np.max(resized_map)
    resized_map *= 255
    resized_map = resized_map.astype(np.uint8)
    cv2.imshow('map', resized_map)
    if cv2.waitKey(1) == ord('q'):
        exit(0)



def visualize_timestep(occupancy_map, particles):
    # global resized_map
    resized_map = cv2.resize(occupancy_map, (800, 800), interpolation=cv2.INTER_NEAREST)
    resized_map -= np.min(resized_map)
    resized_map /= np.max(resized_map)
    resized_map *= 255
    resized_map = resized_map.astype(np.uint8)
    resized_map = cv2.cvtColor(resized_map, cv2.COLOR_GRAY2BGR)
    
    for x in particles:
        print(x)
        cv2.circle(resized_map, (int(x[0] /10), int(x[1]/10 )), 1, (0, 0, 255), -1)
    
    cv2.imshow('map', resized_map)
    if cv2.waitKey(1) == ord('q'):
        exit(0)
    

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    # print(np.min(occupancy_map), np.max(occupancy_map))
    y, x = np.where(occupancy_map == 0)
    chosen_indices = np.random.choice(len(x), num_particles)
    x0_vals = x[chosen_indices].reshape(-1, 1) * 10
    y0_vals = y[chosen_indices].reshape(-1, 1) * 10
    
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    # np.random.seed(501)
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    # print(map_obj.get_map_size_x(), map_obj.get_map_size_y())
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    # viz_map_particles(occupancy_map, X_bar)
    # visualize_timestep(occupancy_map, X_bar)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    thresholded = occupancy_map.copy()
    thresholded[thresholded >= 0] = 1
    thresholded[thresholded < 0] = 0
    
    thresholded *= 255
    thresholded = thresholded.astype(np.uint8)
    
    dtransform = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)
    dtransform -= np.min(dtransform)
    dtransform /= np.max(dtransform)
    
    # resized_map = occupancy_map.copy()
    # resized_map = cv2.resize(occupancy_map, (800, 800), interpolation=cv2.INTER_NEAREST)
    # resized_map -= np.min(resized_map)
    # resized_map /= np.max(resized_map)
    # resized_map *= 255
    # resized_map = resized_map.astype(np.uint8)
    # resized_map = cv2.cvtColor(resized_map, cv2.COLOR_GRAY2BGR)
    vizmap = cv2.resize(occupancy_map, (800, 800), interpolation=cv2.INTER_NEAREST)
    vizmap -= np.min(vizmap)
    vizmap /= np.max(vizmap)
    vizmap *= 255
    vizmap = vizmap.astype(np.uint8)
    vizmap = cv2.cvtColor(vizmap, cv2.COLOR_GRAY2BGR)
    cv2.imshow('dtransform', occupancy_map)
    cv2.waitKey(0)
    
    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        tvizmap = vizmap.copy()
        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s. Measurement type = {}".format(
            time_idx, round(time_stamp, 2), meas_type))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        psum = 0
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            # X_bar_new[m:, :3] = x_t1
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t, tvizmap = sensor_model.beam_range_finder_model(z_t, x_t1, tvizmap, dtransform)
                # print(w_t)
                # psum += w_t
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
            # cv2.circle(vizmap, (int(x_t1[0] / 10), int(x_t1[1] / 10)), 5, (0, 255, 0), -1)
            cv2.arrowedLine(tvizmap, (int(x_t0[0] / 10), int(x_t0[1] / 10)), (int(x_t1[0] / 10), int(x_t1[1] / 10)), (0, 255, 0), 3)
        
        # X_bar_new[:, 3] /= np.sum(X_bar_new[:, 3])
        
        if meas_type == 'L':
            # print(np.sum(X_bar_new[:, 3]))
            cv2.imshow('mapp', tvizmap)
            if cv2.waitKey(1) == ord('q'):
                exit()
        # visualize_timestep(occupancy_map, X_bar_new)
        X_bar = X_bar_new
        u_t0 = u_t1
        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)
