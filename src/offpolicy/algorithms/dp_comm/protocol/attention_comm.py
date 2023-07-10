import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from offpolicy.algorithms.dp_comm.dp_noise import create_noise_generator
import numpy as np

# constrain the vector's norm into C
def norm_constrain(input, C):
    if len(input.shape)==1:
        norm = torch.norm(input, p=2, dim=0)
        return input * (C / norm)
    elif len(input.shape)==3:
        norm = torch.norm(input, p=2, dim=-1)
        return input * (C / norm.unsqueeze(-1))
    else:
        import pdb; pdb.set_trace()
    

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.fc_query = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.fc_key = nn.Linear(in_channels=in_channels, out_channels=in_channels)
        self.fc_value = nn.Linear(in_channels=in_channels, out_channels=in_channels)
        self.in_channels = in_channels
    
    def forward(self, query, key, value):
        N, C, H, W = value.shape 
        q = self.fc_query(query).reshape(N, C, 1)  # .permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H * W)  # .permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H * W)  # .permute(0, 2, 1)
        attention = k.transpose(1, 2) @ q / C ** 0.5
        attention = attention.softmax(dim=1)
        output = v @ attention
        output = output.reshape(N, C, H, W)
        return query + output


class AttentionComm(nn.Module):
    """
    Take (action,state) as input and give a message from i to j

    This is set to be a receiver.
    """
    def __init__(self, input_shape, device, args):
        super(AttentionComm, self).__init__()
        self.args = args
        self.message_dim = args.message_dim
        self.num_agents = args.num_agents
        self.input_shape = input_shape
        self.device = device

        input_attention_shape = input_shape // (self.num_agents - 1)
        self.fc_query = nn.Linear(in_features=input_attention_shape, out_features=args.comm_hidden_dim)
        self.fc_key = nn.Linear(in_features=input_attention_shape, out_features=args.comm_hidden_dim)
        self.fc_value = nn.Linear(in_features=input_attention_shape, out_features=args.comm_hidden_dim)

        self.fc_mean = nn.Linear(args.comm_hidden_dim * (self.num_agents - 1), args.message_dim)
        self.fc_variance = nn.Linear(args.comm_hidden_dim * (self.num_agents - 1), args.message_dim)

        if args.achieve_dp:
            self.achieve_dp = True
            self.sensitivity = args.sensitivity
            self.noise_generator = create_noise_generator(dimension=args.message_dim, epsilon=args.privacy_budget, 
                                                delta=args.delta,
                                                sensitivity=args.sensitivity, method=args.noise_method, episode_length=args.episode_length, episodic_dp=args.episodic_dp)
        else:
            self.achieve_dp = False

        self.to(self.device)

    def forward(self, inputs):
        if(inputs.device!=self.device):
            inputs = inputs.to(self.device)
        if(len(inputs.shape)==1):
            inputs = inputs.reshape(self.num_agents-1, -1)
        else:
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], self.num_agents-1, -1)

        q = self.fc_query(inputs)
        k = self.fc_key(inputs)
        v = self.fc_value(inputs)
        
        if(len(inputs.shape)==2):
            N, DIM = inputs.shape
            attention = k.transpose(0,1) @ q / N ** 0.5
            attention = attention.softmax(dim=0)
            output = v @ attention
            output = output.flatten()
        else: 
            _, _, N, DIM = inputs.shape
            attention = k.transpose(2, 3) @ q / N ** 0.5
            attention = attention.softmax(dim=-1)
            output = v @ attention
            output = output.reshape(output.shape[0], output.shape[1], -1)

        mu, sigma = self.fc_mean(output), self.fc_variance(output) # mean and variance
        # with reparametrization trick
        data_point = mu
        if self.args.gaussian_receiver:
            eps = random.normalvariate(0,1)
            data_point = data_point + eps * sigma
        if self.args.achieve_dp_in_receiver and self.achieve_dp:
            data_point = norm_constrain(data_point, self.sensitivity)
            data_point = data_point + self.noise_generator.noise().to(self.device)
        self.variance = sigma**2
        return data_point
