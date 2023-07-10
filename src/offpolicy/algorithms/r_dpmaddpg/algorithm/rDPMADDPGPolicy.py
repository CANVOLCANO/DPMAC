import numpy as np
import torch
from torch.distributions import OneHotCategorical
from offpolicy.algorithms.r_dpmaddpg.algorithm.r_actor_critic import R_MADDPG_Actor, R_MADDPG_Critic
from offpolicy.utils.util import is_discrete, is_multidiscrete, get_dim_from_space, DecayThenFlatSchedule, soft_update, hard_update, \
    gumbel_softmax, onehot_from_logits, gaussian_noise, avail_choose, to_numpy
from offpolicy.algorithms.base.recurrent_policy import RecurrentPolicy
from offpolicy.algorithms.dp_comm.protocol import create_comm_protocol

from icecream import ic
from termcolor import colored

def dpmac_hook(module, grad_input, grad_output):
    grad_input_msg = grad_input[0]
    grad_input_shape = grad_input_msg.size()
    episode_length, batchsize, dim = grad_input_shape
    
    # compute theoretical variance
    gamma1 = 1.0
    gamma2 = 1.0
    global num_agents
    global C
    global epsilon
    # delta=0.01
    global delta
    global noise_scale
    beta = 0.5    # in (0, 1)
    alpha = np.log(1/delta) / (epsilon *  (1 - beta))
    square_std = 14 * (gamma1**2) * gamma2 * float(num_agents) * (C**2) * alpha / (beta * epsilon)
    std = np.sqrt(square_std)

    # perform norm clip
    grad_input_msg = grad_input_msg.view(episode_length*batchsize, -1)
    grad_input_norm = torch.norm(grad_input_msg, p=2, dim=1)
    clip_bound = C / (batchsize*episode_length)
    clip_coef = clip_bound / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_input_msg = clip_coef * grad_input_msg

    # add noise
    noise = torch.normal(mean=0, std=std, size=grad_input_msg.size()).to(grad_input_msg.device) * noise_scale
    grad_input_msg = grad_input_msg + noise
    
    # save
    grad_input_new = [grad_input_msg.view(grad_input_shape)]
    for i in range(len(grad_input)-1):
        grad_input_new.append(grad_input[i+1])
    
    return tuple(grad_input_new)

class R_DPMADDPGPolicy(RecurrentPolicy):
    """
    Recurrent MADDPG Policy Class to wrap actor/critic and compute actions. See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param target_noise: (int) std of target smoothing noise to add for MATD3 (applies only for continuous actions)
    :param td3: (bool) whether to use MATD3 or MADDPG.
    :param train: (bool) whether the policy will be trained.
    """
    def __init__(self, config, policy_config, target_noise=None, td3=False, train=True):
        self.config = config
        self.device = config['device']
        self.args = self.config["args"]
        self.tau = self.args.tau
        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self.prev_act_inp = self.args.prev_act_inp

        self.central_obs_dim, self.central_act_dim = policy_config["cent_obs_dim"], policy_config["cent_act_dim"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.hidden_size = self.args.hidden_size
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        actor_class = R_MADDPG_Actor
        critic_class = R_MADDPG_Critic
       
        actor_input_dim = self.obs_dim + self.args.message_dim # actor with communication

        self.actor = actor_class(self.args, actor_input_dim, self.act_dim, self.device, take_prev_action=self.prev_act_inp)
        self.critic = critic_class(self.args, self.central_obs_dim, self.central_act_dim, self.device)

        self.target_actor = actor_class(self.args, actor_input_dim, self.act_dim, self.device, take_prev_action=self.prev_act_inp)
        self.target_critic = critic_class(self.args, self.central_obs_dim, self.central_act_dim, self.device)

        # sync the target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        ##### dp communication #####
        # communication
        self.num_agents = policy_config["num_agents"]
        self.agent_id = policy_config["agent_id"]

        if self.args.msg_include_id: # include agent id
            # sender input shape: local obs + local action + receiver id
            sender_input_shape =  self.obs_dim + self.act_dim  + 1
            # receiver input shape: other agents num * (message dim + sender id)
            receiver_input_shape = ( self.num_agents -1 ) * (self.args.message_dim + 1 )
        else:
            sender_input_shape =  self.obs_dim + self.act_dim
            receiver_input_shape = ( self.num_agents - 1 ) * (self.args.message_dim)

        self.sender = create_comm_protocol(self.args.sender_method , sender_input_shape, self.device,  self.args).to(self.device)
        self.receiver = create_comm_protocol(self.args.receiver_method, receiver_input_shape,  self.device, self.args).to(self.device)

        ### Register hook for privacy ###
        use_privacy_hook = False
        if use_privacy_hook:
            if self.args.achieve_dp:
                global num_agents
                global C
                global epsilon
                global delta
                global noise_scale
                num_agents = self.num_agents
                C = self.args.sensitivity / 2.0
                epsilon = self.args.privacy_budget
                delta = self.args.DP_delta
                noise_scale = self.args.noise_scale
                self.sender.fc1.register_backward_hook(dpmac_hook)
        ### end ###
        ##### end #####

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

            self.sender_optimizer = torch.optim.Adam(self.sender.parameters(), lr=self.args.sender_lr, weight_decay=self.weight_decay)
            self.receiver_optimizer = torch.optim.Adam(self.receiver.parameters(), lr=self.args.receiver_lr, weight_decay=self.weight_decay)
            if self.discrete:
                # eps greedy exploration
                self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish,
                                                         self.args.epsilon_anneal_time, decay="linear")
        self.target_noise = target_noise

    def get_actions(self, obs, message, prev_actions, rnn_states, available_actions=None, t_env=None, explore=False, use_target=False, use_gumbel=False):
        """
        See parent class.
        :param use_target: (bool) whether to use the target actor or live actor.
        :param use_gumbel: (bool) whether to apply gumbel softmax on the actions.
        """
        assert prev_actions is None or len(obs.shape) == len(prev_actions.shape)
        # obs is either an array of shape (batch_size, obs_dim) or (seq_len, batch_size, obs_dim)
        if len(obs.shape) == 2:
            batch_size = obs.shape[0]
            no_sequence = True
            use_tensor = True if  message.__class__==torch.Tensor and obs.__class__==torch.Tensor else False
            if message.__class__==torch.Tensor and obs.__class__==np.ndarray:
                obs = torch.tensor(obs).to(message.device)
                use_tensor = True
            elif obs.__class__== torch.Tensor and message.__class__== np.ndarray:
                message = torch.tensor(message).to(obs.device)
                use_tensor = True
            if use_tensor:
                obs = torch.cat([obs, message], dim=-1) # concat the msg together
            else:
                obs = np.concatenate([obs, message], axis=-1)
        else: # sequence
            batch_size = obs.shape[1]
            no_sequence = False
            use_tensor = True if  message.__class__==torch.Tensor and obs.__class__==torch.Tensor else False
            if message.__class__==torch.Tensor and obs.__class__==np.ndarray:
                obs = torch.tensor(obs).to(message.device)
                use_tensor = True
            elif obs.__class__== torch.Tensor and message.__class__== np.ndarray:
                message = torch.tensor(message).to(obs.device)
                use_tensor = True
            
            if use_tensor:
                obs = torch.cat([obs, message], dim=-1) # concat the msg together
            else:
                obs = np.concatenate([obs, message], axis=-1)

        eps = None
        if obs.__class__==torch.Tensor:
            obs = obs.float()
        if use_target:
            actor_out, new_rnn_states = self.target_actor(obs, prev_actions, rnn_states)
        else:
            actor_out, new_rnn_states = self.actor(obs, prev_actions, rnn_states) # in po_spread actor_out torch.Size([1, 5])

        if self.discrete:
            if self.multidiscrete:
                if use_gumbel or (use_target and self.target_noise is not None):
                    onehot_actions = list(map(lambda a: gumbel_softmax(a, hard=True, device=self.device), actor_out))
                    actions = torch.cat(onehot_actions, dim=-1)
                elif explore:
                    onehot_actions = list(map(lambda a: gumbel_softmax(a, hard=True, device=self.device), actor_out))
                    onehot_actions = torch.cat(onehot_actions, dim=-1)
                    assert no_sequence, "Doesn't make sense to do exploration on a sequence!"
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    take_random = (rand_numbers < eps).astype(int).reshape(-1, 1)
                    # random actions sample uniformly from action space
                    random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample() for i in range(len(self.act_dim))]
                    random_actions = torch.cat(random_actions, dim=1)
                    actions = (1 - take_random) * to_numpy(onehot_actions) + take_random * to_numpy(random_actions)
                else:
                    onehot_actions = list(map(onehot_from_logits, actor_out))
                    actions = torch.cat(onehot_actions, dim=-1)
  
            else:
                if use_gumbel or (use_target and self.target_noise is not None):
                    actions = gumbel_softmax(actor_out, available_actions, hard=True, device=self.device)  # gumbel has a gradient 
                elif explore:
                    onehot_actions = gumbel_softmax(actor_out, available_actions, hard=True, device=self.device)  # gumbel has a gradient                    
                    assert no_sequence, "Cannot do exploration on a sequence!"
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    # random actions sample uniformly from action space
                    logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                    random_actions = OneHotCategorical(logits=logits).sample().numpy()
                    take_random = (rand_numbers < eps).astype(int)
                    actions = (1 - take_random) * to_numpy(onehot_actions) + take_random * random_actions
                else:
                    actions = onehot_from_logits(actor_out, available_actions)  # no gradient

        else:
            if explore:
                assert no_sequence, "Cannot do exploration on a sequence!"
                actions = gaussian_noise(actor_out.shape, self.args.act_noise_std).to(actor_out.device) + actor_out
            elif use_target and self.target_noise is not None:
                assert isinstance(self.target_noise, float)
                actions = gaussian_noise(actor_out.shape, self.target_noise).to(actor_out.device) + actor_out
            else:
                actions = actor_out
        return actions, new_rnn_states, eps

    def init_hidden(self, num_agents, batch_size):
        """See parent class."""
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(num_agents, batch_size, self.hidden_size)

    def get_random_actions(self, obs, message,  available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]
        if self.discrete:
            if self.multidiscrete:
                random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i in
                                    range(len(self.act_dim))]
                random_actions = np.concatenate(random_actions, axis=-1)
            else:
                if available_actions is not None:
                    logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                    random_actions = OneHotCategorical(logits=logits).sample().numpy()
                else:
                    random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()
        else:
            random_actions = np.random.uniform(self.act_space.low, self.act_space.high, size=(batch_size, self.act_dim))

        return random_actions

    def soft_target_updates(self):
        """Soft update the target networks through a Polyak averaging update."""
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

    def hard_target_updates(self):
        """Hard update target networks by copying the weights of the live networks."""
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)
