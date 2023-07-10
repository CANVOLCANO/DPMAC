import numpy as np

class LaplaceNoise:
    def __init__(self, dimension, epsilon=0.5, sensitivity=1.):
        self.epsilon = epsilon
        self.dimension = dimension
        self.scale = sensitivity / epsilon 
    

    def generate(self):
        noise =  np.random.laplace(scale=self.scale, size=self.dimension)
        return noise