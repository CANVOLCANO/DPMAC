import numpy as np
import torch
from termcolor import colored
from icecream import ic


class GaussianNoise:
    def __init__(self, dimension, epsilon, delta, sensitivity, n_agent, episode_length, episodic_dp):
        '''
        if n_agent>1, noise will be such that receiver achieves (eps, delta)-DP by composition lemma
        '''
        self.epsilon = epsilon
        self.delta = delta
        self.dimension = dimension
        self.sensitivity = sensitivity
        self.sampling_rate1 = 1.0/episode_length 
        self.sampling_rate2 = 1.0 
        self.N = 2.0 # num of agents to send messages
        constant = 14.0
        self.n_agent = n_agent
        self.beta = 1/2
        if n_agent>1: # if achieve dp in receiver instead of sender, need privacy composition lemma
            self.epsilon /= self.n_agent
            self.delta /= self.n_agent
        self.alpha = np.log(1/self.delta) / (self.epsilon*(1-self.beta)) + 1
        sigma_square = constant * self.sampling_rate2 * self.sampling_rate1**2 * self.N * self.alpha * sensitivity**2 / (self.beta*self.epsilon)
        if episodic_dp:
            sigma_square *= episode_length
        self.std = np.sqrt(sigma_square)
        print(colored(" noise std: "+str(self.std), 'magenta' ) )


    def noise(self):
        noise =  np.random.normal(0, self.std, self.dimension)
        return torch.tensor(noise, dtype=torch.float32)
