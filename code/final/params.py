import numpy as np

random_seed = np.random.randint(0, 1000000)
np.random.seed(random_seed)

params = {
    "n_particles" : 10000,
    "random_seed" : random_seed,
    "min_sampling_probability" : 0.1,
    "motion_model" : {
        "alpha1" : 0.0005,
        "alpha2" : 0.0005,
        "alpha3" : 0.005,
        "alpha4" : 0.005,    
    },    
    "sensor_model": {
        "max_range": 8000,   
        "subsampling" : 1,
        "min_probability" : 0.35,
        "sigma_hit": 150,
        "lambda_short": 0.1,
        "z_hit": 1,
        "z_short": 0.1,
        "z_max": 0.05,
        "z_rand": 500
    },
    "sampling_params": {
        "decay_rate": 0.0001,
        "decay_steps": 1000,
        "min_particles": 500,
        "resampling_noise" : {
            "x" : 2,
            "y" : 2,
            "theta" : 0.05,
        }
    },
    "map_path" : '/home/praveenvnktsh/slam/data/map/wean.dat',
    "log_path" : '/home/praveenvnktsh/slam/data/log/robotdata1.log',
    "precomputed_raycast_file": "/home/praveenvnktsh/slam/code/new/precomputed_cast_8183_35_5.npy",
}

precomputed_cast = np.load(params['precomputed_raycast_file'])
