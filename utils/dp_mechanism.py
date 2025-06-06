import numpy as np



def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr  * clip / dataset_size


def Gaussian_Simple(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon



