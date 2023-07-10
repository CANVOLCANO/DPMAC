import numpy as np
import torch
import torch.nn.functional as F
import time
import math
from offpolicy.utils.util import to_torch
from offpolicy.runner.rnn.base_runner import RecRunner

from icecream import ic


class MPERunner(RecRunner):
    """Runner class for Multiagent Particle Envs (MPE). See parent class for more information."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        if(not self.use_communication):
            self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        else:
            self.collecter = self.shared_collect_communication_rollout if self.share_policy else self.separated_collect_communication_rollout
        # fill replay buffer with random actions
        num_warmup_episodes = max(
            (self.batch_size, self.args.num_random_episodes))
        self.start = time.time()
        self.log_clear()

    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []
        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter(
                explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)
        self.log_env(eval_infos, suffix="_")

    # for mpe-simple_spread and mpe-simple_reference
    @torch.no_grad()
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]
        env = self.env if training_episode or warmup else self.eval_env
        obs = env.reset()
        rnn_states_batch = np.zeros(
            (self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros(
            (self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)
        # initialize variables to store episode information.
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents,
                                      policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents,
                                            policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents,
                                       policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros(
            (self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones(
            (self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones(
            (self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}
        t = 0
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                            last_acts_batch,
                                                            rnn_states_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                     last_acts_batch,
                                                                     rnn_states_batch,
                                                                     t_env=self.total_env_steps,
                                                                     explore=explore)
            acts_batch = acts_batch if isinstance(
                acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(
                rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch
            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            if training_episode:
                self.total_env_steps += self.num_envs
            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(
                dones_env) or t == self.episode_length - 1
            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1
            obs = next_obs
            if terminate_episodes:
                break
        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)
        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts)
        average_episode_rewards = np.mean(
            np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards
        return env_info

    # for mpe-simple_speaker_listener

    @torch.no_grad()
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env
        obs = env.reset()
        rnn_states = np.zeros(
            (self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32)
        last_acts = {p_id: np.zeros((self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros((self.episode_length, self.num_envs, len(
            self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones((self.episode_length, self.num_envs, len(
            self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones(
            (self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}
        t = 0
        while t < self.episode_length:
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                agent_obs = np.stack(obs[:, agent_id])
                share_obs = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
                                                                                                -1).astype(np.float32)
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(agent_obs,
                                                         last_acts[p_id][:, 0],
                                                         rnn_states[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                               last_acts[p_id],
                                                               rnn_states[agent_id],
                                                               sample=explore)
                    else:
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                               last_acts[p_id].squeeze(
                                                                   axis=0),
                                                               rnn_states[agent_id],
                                                               t_env=self.total_env_steps,
                                                               explore=explore)
                # update rnn hidden state
                rnn_states[agent_id] = rnn_state if isinstance(
                    rnn_state, np.ndarray) else rnn_state.cpu().detach().numpy()
                last_acts[p_id] = np.expand_dims(act, axis=1) if isinstance(
                    act, np.ndarray) else np.expand_dims(act.cpu().detach().numpy(), axis=1)
                episode_obs[p_id][t] = agent_obs
                episode_share_obs[p_id][t] = share_obs
                if(act.__class__ == torch.Tensor and act.device != 'cpu'):
                    episode_acts[p_id][t] = act.cpu()
                else:
                    episode_acts[p_id][t] = act
            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for p_id in self.policy_ids:
                    env_act.append(last_acts[p_id][i, 0])
                env_acts.append(env_act)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(
                dones_env) or t == self.episode_length - 1
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                episode_rewards[p_id][t] = np.expand_dims(
                    rewards[:, agent_id], axis=1)
                episode_dones[p_id][t] = np.expand_dims(
                    dones[:, agent_id], axis=1)
                episode_dones_env[p_id][t] = dones_env
            obs = next_obs
            t += 1
            if training_episode:
                self.total_env_steps += self.num_envs
            if terminate_episodes:
                break
        for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
            episode_obs[p_id][t] = np.stack(obs[:, agent_id])
            episode_share_obs[p_id][t] = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
                                                                                                             -1).astype(np.float32)

        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(self.num_envs, episode_obs, episode_share_obs, episode_acts,
                               episode_rewards, episode_dones, episode_dones_env, episode_avail_acts)
        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(
                np.mean(np.sum(episode_rewards[p_id], axis=0)))
        env_info['average_episode_rewards'] = np.mean(average_episode_rewards)
        return env_info

    @torch.no_grad()
    def init_communicate(self, training_episode=True):
        """
        The initialized communication without any env input.

        Intuitively, make agents know each other.
        """
        assert self.num_envs == 1, "communication alg not support parallel env. if you wanna use, just implement it."
        self.data_center = dict()  # refresh
        # data center collects the msg without integration
        for agent_i, p_i in zip(self.agent_ids, self.policy_ids):
            policy_i = self.policies[p_i]
            for agent_j, p_j in zip(self.agent_ids, self.policy_ids):
                policy_j = self.policies[p_j]
                if(agent_i == agent_j):
                    continue
                if self.args.msg_include_id:
                    # minus 1 for concat of agent id
                    nothing = torch.zeros(policy_i.sender.input_shape - 1)
                    input = torch.cat([nothing, torch.tensor([agent_j])]).to(
                        policy_i.sender.device)  # send to j
                else:
                    input = torch.zeros(policy_i.sender.input_shape).to(
                        policy_i.sender.device)
                message_ij = policy_i.sender(input, not training_episode).squeeze(0)  # mean
                self.data_center[(agent_i, agent_j)] = message_ij
        self.message_received = self.message_integration()

    @torch.no_grad()
    def message_integration(self):
        """
        Integrate messages from other agents into one message
        used for DPMAC
        """
        self.integration_center = dict()  # refresh
        self.concated_msg_center = dict()
        # integration center collects the msg with integration
        messages_received = []
        device = self.policies[self.policy_ids[0]].receiver.device
        for agent_i, p_i in zip(self.agent_ids, self.policy_ids):
            message_i = []
            policy_i = self.policies[p_i]
            for agent_j, p_j in zip(self.agent_ids, self.policy_ids):
                if agent_i == agent_j:
                    continue
                # collect
                message_ji = self.data_center[(agent_j, agent_i)].to(device)
                if self.args.msg_include_id:
                    message_ji = torch.cat([message_ji, torch.tensor(
                        [agent_j]).to(device)])  # add sender id
                message_i.append(message_ji)
            message_i = torch.cat(message_i)
            # store the input of the receiver
            self.concated_msg_center[agent_i] = message_i.clone()
            message_i = policy_i.receiver(message_i)
            self.integration_center[agent_i] = message_i
            messages_received.append(message_i)
        return messages_received

    @torch.no_grad()
    def communicate(self, observations, actions, training_episode=True):
        """
        Make one step's communication
        """
        self.data_center = dict()  # refresh
        for agent_i, p_i, observation_i, action_i in zip(self.agent_ids, self.policy_ids, observations, actions):
            policy_i = self.policies[p_i]
            # send message from i to j
            for agent_j, p_j in zip(self.agent_ids, self.policy_ids):
                if agent_j == agent_i:
                    continue
                if self.args.msg_include_id:
                    input = np.concatenate(
                        [observation_i, action_i, [agent_j]], axis=0)
                else:
                    input = np.concatenate([observation_i, action_i], axis=0)
                input = to_torch(input).unsqueeze(0).float()
                message_ij = policy_i.sender(input, not training_episode).squeeze(0)
                self.data_center[(agent_i, agent_j)] = message_ij
        self.message_integration()

    # for mpe-simple_speaker_listener

    @torch.no_grad()
    def separated_collect_communication_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Used for alg with communication.
        Collect a rollout and store it in the buffer. Each agent has its own policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env
        obs = env.reset()
        rnn_states = np.zeros(
            (self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32)
        last_acts = {p_id: np.zeros((self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, len(
            self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros((self.episode_length, self.num_envs, len(
            self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones((self.episode_length, self.num_envs, len(
            self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones(
            (self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}
        episode_messages = {p_id: np.zeros((self.episode_length+1, self.num_envs, len(
            self.policy_agents[p_id]), self.args.message_dim), dtype=np.float32) for p_id in self.policy_ids}
        t = 0
        # init communication msg
        self.init_communicate(training_episode)
        while t < self.episode_length:
            # 1. take action
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                agent_obs = np.stack(obs[:, agent_id])
                share_obs = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(
                    self.num_envs, -1).astype(np.float32)
                message_i = self.integration_center[agent_id]  # use msg
                # message_i_for_receiver = self.concated_msg_center[agent_id]
                message_i = message_i.unsqueeze(0)
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs, message_i)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(agent_obs, message_i,
                                                         last_acts[p_id][:, 0],
                                                         rnn_states[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(agent_obs, message_i,
                                                               last_acts[p_id],
                                                               rnn_states[agent_id],
                                                               sample=explore)
                    else:
                        act, rnn_state, _ = policy.get_actions(agent_obs, message_i,
                                                               last_acts[p_id].squeeze(
                                                                   axis=0),
                                                               rnn_states[agent_id],
                                                               t_env=self.total_env_steps,
                                                               explore=explore)
                # update rnn hidden state
                rnn_states[agent_id] = rnn_state if isinstance(
                    rnn_state, np.ndarray) else rnn_state.cpu().detach().numpy()
                last_acts[p_id] = np.expand_dims(act, axis=1) if isinstance(
                    act, np.ndarray) else np.expand_dims(act.cpu().detach().numpy(), axis=1)
                episode_obs[p_id][t] = agent_obs
                episode_share_obs[p_id][t] = share_obs
                if(act.__class__ == torch.Tensor and act.device != 'cpu'):
                    episode_acts[p_id][t] = act.cpu()
                else:
                    episode_acts[p_id][t] = act
                if(message_i.__class__ == torch.Tensor and message_i.device != 'cpu'):
                    episode_messages[p_id][t] = message_i.cpu()
                else:
                    episode_messages[p_id][t] = message_i
            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for p_id in self.policy_ids:
                    env_act.append(last_acts[p_id][i, 0])
                env_acts.append(env_act)
            # 2. send message
            sampled_obs = np.zeros_like(obs)
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                sampled_obs[0][agent_id] = episode_obs[p_id][np.random.randint(t+1, size=1)]
            if self.algorithm_name == 'rdpmaddpg' and self.achieve_dp:
                sent_obs = sampled_obs  
            else:
                sent_obs = obs
            # self.communicate(obs.squeeze(0), env_acts[0], training_episode)
            self.communicate(sent_obs.squeeze(0), env_acts[0], training_episode)
            # 3. get integration message
            messages_received = self.message_received
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(
                dones_env) or t == self.episode_length - 1
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                episode_rewards[p_id][t] = np.expand_dims(
                    rewards[:, agent_id], axis=1)
                episode_dones[p_id][t] = np.expand_dims(
                    dones[:, agent_id], axis=1)
                episode_dones_env[p_id][t] = dones_env
            obs = next_obs
            # ic('in mpe runner', next_obs.shape) # (1, n_agents, dim_per_obs)
            t += 1
            if training_episode:
                self.total_env_steps += self.num_envs
            if terminate_episodes:
                break
        for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
            episode_obs[p_id][t] = np.stack(obs[:, agent_id])
            episode_share_obs[p_id][t] = np.concatenate([obs[0, i] for i in range(
                self.num_agents)]).reshape(self.num_envs, -1).astype(np.float32)
            message_i = self.integration_center[agent_id]  # use msg
            message_i = message_i.unsqueeze(0)
            if(message_i.__class__ == torch.Tensor and message_i.device != 'cpu'):
                episode_messages[p_id][t] = message_i.cpu()
            else:
                episode_messages[p_id][t] = message_i
        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(self.num_envs, episode_obs, episode_share_obs, episode_acts, episode_rewards, episode_dones,
                               episode_dones_env, episode_avail_acts, episode_messages)
        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(
                np.mean(np.sum(episode_rewards[p_id], axis=0)))
        env_info['average_episode_rewards'] = np.mean(average_episode_rewards)
        return env_info


    @torch.no_grad()
    def shared_collect_communication_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]
        env = self.env if training_episode or warmup else self.eval_env
        obs = env.reset()
        rnn_states_batch = np.zeros(
            (self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros(
            (self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)
        # initialize variables to store episode information.
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents,
                                      policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents,
                                            policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents,
                                       policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros(
            (self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones(
            (self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones(
            (self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}
        # storing the messgae
        episode_messages = {p_id: None for p_id in self.policy_ids}
        t = 0
        self.init_communicate(training_episode)
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch, message)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch, message,
                                                            last_acts_batch,
                                                            rnn_states_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                     last_acts_batch,
                                                                     rnn_states_batch,
                                                                     t_env=self.total_env_steps,
                                                                     explore=explore)
            acts_batch = acts_batch if isinstance(
                acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(
                rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch
            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            if training_episode:
                self.total_env_steps += self.num_envs
            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(
                dones_env) or t == self.episode_length - 1
            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1
            obs = next_obs
            if terminate_episodes:
                break
        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)
        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts)
        average_episode_rewards = np.mean(
            np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards
        return env_info


    def log_clear(self):
        """See parent class."""
        self.env_infos = {}
        self.env_infos['average_episode_rewards'] = []
