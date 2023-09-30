import numpy as np
from numba import njit

# Used in p_max and p_rand, optionally in ray casting
_max_range = 8183

# Used for thresholding obstacles of the occupancy map
_min_probability = 0.35

resolution = 5

def load_ogm():
    mappath = '/home/praveenvnktsh/slam/data/map/wean.dat'
    occupancy_map = np.genfromtxt(mappath, skip_header=7)
    occupancy_map[occupancy_map < 0] = -1
    occupancy_map[occupancy_map > 0] = 1 - occupancy_map[occupancy_map > 0]
    occupancy_map = np.flipud(occupancy_map)
    
    return occupancy_map

@njit
def raycast(occupancy_map, x):
    true_measurements = np.zeros((360, 2))
    for idx, angle in enumerate(np.linspace(0, 360, 360)):
        x_t = x[0] #+ laser_frame[0]
        y_t = x[1] #+ laser_frame[1]
        # print(_max_range, )
        angle = np.deg2rad(angle)
        for  dist in (range(0, _max_range + resolution, resolution)):
            x_t = x[0] + dist * np.cos(angle)
            y_t = x[1] + dist * np.sin(angle)

            if x_t >= 0 and x_t <= occupancy_map.shape[1] * 10 and y_t >= 0 and y_t <= occupancy_map.shape[0] * 10:
                if occupancy_map[int(y_t/10), int(x_t/10)] > _min_probability or occupancy_map[int(y_t/10), int(x_t/10)] < 0:
                    true_measurements[idx][0] = (np.sqrt((x_t - x[0]) ** 2 + (y_t - x[1]) ** 2))
                    true_measurements[idx][1] = angle
                    # true_measurements[idx] = dist
                    
                    break
    return true_measurements

ogm = load_ogm()
@njit
def compute_cast():
    # x, y, theta, measurement
    
    array = np.zeros((800, 800, 360,), dtype=np.float64)
    for x in range(ogm.shape[0]):
        print(x)
        for y in range(ogm.shape[1]):
            true_measurements = raycast(ogm, np.array([x*10, y*10]))
            for i in range(len(true_measurements)):
                msmt, theta = true_measurements[i]
                theta = np.rad2deg(theta)
                array[x, y, int(theta)] = msmt

    return array



np.save(f"precomputed_cast_{_max_range}_{round(_min_probability * 100)}_{resolution}.npy", compute_cast())
print("Done")
    

