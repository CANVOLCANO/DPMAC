from .gaussian import GaussianNoise
from .laplace import LaplaceNoise

def create_noise_generator(dimension, epsilon, delta, sensitivity, method, n_agent=1, episode_length=None, episodic_dp=0):
    if method=='gauss':
        return GaussianNoise(dimension, epsilon, delta, sensitivity, n_agent, episode_length, episodic_dp)
    elif method=='laplace':
        return LaplaceNoise(dimension, epsilon, sensitivity)
