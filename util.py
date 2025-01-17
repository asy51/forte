import numpy as np

def sample_gaussian(n=1_000, mean=[0,0], std=[1,1]):
    mean = np.array(mean)
    std = np.array(std)
    samples = np.random.randn(n, len(mean)) * std + mean
    return samples