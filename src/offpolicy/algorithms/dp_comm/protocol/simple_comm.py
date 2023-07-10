import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from offpolicy.algorithms.dp_comm.dp_noise import create_noise_generator
import numpy as np
from icecream import ic


def norm_constrain(input, C):
    if len(input.shape)==1:
        norm = torch.norm(input, p=2, dim=0)
        return input * (C / norm)
    elif len(input.shape)==2 or len(input.shape)==3:
        norm = torch.norm(input, p=2, dim=-1)
        return input * (C / norm.unsqueeze(-1))
    else:
        raise ValueError("input dimension is not supported")

class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class SimpleComm(nn.Module):
    """
    Take (action,state) as input and give a message from i to j
    """
    def __init__(self, input_shape, device, args):
        super(SimpleComm, self).__init__()
        self.args = args
        self.message_dim = args.message_dim
        self.input_shape = input_shape
        self.device = device

        self.fc1 = nn.Linear(self.input_shape, args.comm_hidden_dim)
        self.fc_mean = nn.Linear(args.comm_hidden_dim, args.message_dim)
        self.fc_variance = nn.Linear(args.comm_hidden_dim, args.message_dim)

        if args.achieve_dp:
            self.achieve_dp = True
            self.sensitivity = args.sensitivity
            self.noise_generator = create_noise_generator(dimension=args.message_dim, epsilon=args.privacy_budget,
                                                delta=args.delta,
                                                sensitivity=args.sensitivity, method=args.noise_method, n_agent=1, episode_length=args.episode_length, episodic_dp=args.episodic_dp)
        else:
            self.achieve_dp = False
        self.to(self.device)

    def forward(self, inputs, evaluation):
        if(inputs.device!=self.device):
            inputs = inputs.to(self.device)
        x = F.relu(self.fc1(inputs))
        mu, log_sigma = self.fc_mean(x), torch.tanh(self.fc_variance(x)) # mean and variance
        sigma = log_sigma.exp()
        try:
            self.distribution = Normal(mu, sigma)
        except:
            ones = torch.ones_like(mu).to(self.device)
            self.distribution = Normal(mu, ones)

        # with reparametrization trick
        eps = random.normalvariate(0,1)
        self.variance = sigma.pow(2)
        data_point = mu + sigma * eps

        if self.achieve_dp and self.args.achieve_dp_in_sender:
            if evaluation or self.args.dpmac_privacy_priori:
                data_point = norm_constrain(data_point, self.sensitivity)
                data_point = data_point + self.noise_generator.noise().to(self.device)
            
        return data_point
