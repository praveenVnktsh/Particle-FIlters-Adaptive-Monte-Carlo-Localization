'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        pass
    
    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar[:, 3] = np.exp(X_bar[:, 3] - np.max(X_bar[:, 3]))
        # print(X_bar[:, 3])
        num_particles = np.shape(X_bar)[0]
        r1 = np.random.uniform(0, 1/num_particles)
        all_weights = X_bar[:,3]/np.sum(X_bar[:,3])
        c1 = all_weights[0]
        i=0
        new_particles = []
        for m in range(0, num_particles):
            u1 = r1 + (m)*(1/num_particles)
            while u1>c1:
                i += 1
                c1 += all_weights[i]
            new_particles.append(X_bar[i,:])

        X_bar_resampled = np.vstack(new_particles)
        return X_bar_resampled
