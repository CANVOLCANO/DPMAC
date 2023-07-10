import argparse
from email.policy import default
from random import choice, choices
from git import typ
from termcolor import colored


def get_config():
    parser = argparse.ArgumentParser(
        description="MARL", formatter_class=argparse.RawDescriptionHelpFormatter)
    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="rdpmaddpg", choices=["rdpmaddpg"])
    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True)
    parser.add_argument('--n_training_threads', type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument('--n_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for training rollout")
    parser.add_argument('--n_eval_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument('--num_env_steps', type=int,
                        default=1000000, help="Number of env steps to train for")
    parser.add_argument('--user_name', type=str, default="Anonymous")
    # env parameters
    parser.add_argument('--env_name', type=str, default="StarCraft2")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--use_discrete_action", default=True, action='store_false', help="whether to use discrete action (specify it if not use !!! )")
    parser.add_argument("--po_spread_view_num", default=3, type=int, help="the num of agents and landmarks that can be observed by one agent, in po_spread environment")
    parser.add_argument("--po_spread_view_radius", default=1.2, type=float, help="the radius that can be observed by one agent, in po_spread environment")
    parser.add_argument("--pp_view_radius", default=1.2, type=float, help="the radius that can be observed by one agent, in pp environment")
    # replay buffer parameters
    parser.add_argument('--episode_length', type=int,
                        default=25, help="Max length for any episode")
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help="Max # of transitions that replay buffer can contain")
    parser.add_argument('--use_reward_normalization', choices=[0,1], type=int,
                        default=1, help="Whether to normalize rewards in replay buffer")
    parser.add_argument('--use_popart', action='store_true', default=False,
                        help="Whether to use popart to normalize the target loss")
    parser.add_argument('--popart_update_interval_step', type=int, default=2,
                        help="After how many train steps popart should be updated")
    # prioritized experience replay
    parser.add_argument('--use_per', action='store_true', default=False,
                        help="Whether to use prioritized experience replay")
    parser.add_argument('--per_nu', type=float, default=0.9,
                        help="Weight of max TD error in formation of PER weights")
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help="Alpha term for prioritized experience replay")
    parser.add_argument('--per_eps', type=float, default=1e-6,
                        help="Eps term for prioritized experience replay")
    parser.add_argument('--per_beta_start', type=float, default=0.4,
                        help="Starting beta term for prioritized experience replay")
    # network parameters
    parser.add_argument("--use_centralized_Q", action='store_false',
                        default=True, help="Whether to use centralized Q function")
    parser.add_argument('--share_policy', default=0, choices=[0,1],
                        type=int, help="Whether agents share the same policy")
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument('--use_feature_normalization', action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument('--use_orthogonal', action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    # recurrent parameters
    parser.add_argument('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous actions as part of its input")
    parser.add_argument("--use_rnn_layer", action='store_true',
                        default=False, help='Whether to use a recurrent policy')
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument('--data_chunk_length', type=int, default=80,
                        help="Time length of chunks used to train via BPTT")
    parser.add_argument('--burn_in_time', type=int, default=0,
                        help="Length of burn in time for RNN training, see R2D2 paper")
    # attn parameters
    parser.add_argument("--attn", action='store_true', default=False)
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True)
    parser.add_argument("--use_cat_self", action='store_false', default=True)
    # optimizer parameters
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    # algo common parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of buffer transitions to train on at once")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for env")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help='max norm of gradients (default: 1.0)')
    parser.add_argument('--use_huber_loss', action='store_true',
                        default=False, help="Whether to use Huber loss for critic update")
    parser.add_argument("--huber_delta", type=float, default=10.0)
    # soft update parameters
    parser.add_argument('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Polyak update rate")
    # hard update parameters
    parser.add_argument('--hard_update_interval_episode', type=int, default=200,
                        help="After how many episodes the lagging target should be updated")
    parser.add_argument('--hard_update_interval', type=int, default=200,
                        help="After how many timesteps the lagging target should be updated")
    # rmatd3 parameters
    parser.add_argument("--target_action_noise_std", default=0.2, help="Target action smoothing noise for matd3")
    # rmasac parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Initial temperature")
    parser.add_argument('--target_entropy_coef', type=float,
                        default=0.5, help="Initial temperature")
    parser.add_argument('--automatic_entropy_tune', action='store_false',
                        default=True, help="Whether use a centralized critic")
    # qmix parameters
    parser.add_argument('--use_double_q', action='store_false',
                        default=True, help="Whether to use double q learning")
    parser.add_argument('--hypernet_layers', type=int, default=2,
                        help="Number of layers for hypernetworks. Must be either 1 or 2")
    parser.add_argument('--mixer_hidden_dim', type=int, default=32,
                        help="Dimension of hidden layer of mixing network")
    parser.add_argument('--hypernet_hidden_dim', type=int, default=64,
                        help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2")
    # exploration parameters
    parser.add_argument('--num_random_episodes', type=int, default=5,
                        help="Number of episodes to add to buffer with purely random actions")
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_finish', type=float, default=0.05,
                        help="Ending value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_anneal_time', type=int, default=50000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--act_noise_std', type=float,
                        default=0.1, help="Action noise")
    # train parameters
    parser.add_argument('--actor_train_interval_step', type=int, default=2,
                        help="After how many critic updates actor should be updated")
    parser.add_argument('--train_interval_episode', type=int, default=1,
                        help="Number of env steps between updates to actor/critic")
    parser.add_argument('--train_interval', type=int, default=100,
                        help="Number of episodes between updates to actor/critic")
    parser.add_argument("--use_value_active_masks",
                        action='store_true', default=False)
    # eval parameters
    parser.add_argument('--use_eval', action='store_false',
                        default=True, help="Whether to conduct the evaluation")
    parser.add_argument('--eval_interval', type=int,  default=10000,
                        help="After how many episodes the policy should be evaled")
    parser.add_argument('--num_eval_episodes', type=int, default=32,
                        help="How many episodes to collect for each eval")
    # save parameters
    parser.add_argument('--save_interval', type=int, default=100000,
                        help="After how many episodes of training the policy model should be saved")
    # log parameters
    parser.add_argument('--log_interval', type=int, default=1000,
                        help="After how many episodes of training the policy model should be saved")
    # pretained parameters
    parser.add_argument("--model_dir", type=str, default=None)
    parser = private_communication_parser(parser)
    return parser
    

def private_communication_parser(parser):
    ##### private communication #####
    parser.add_argument("--use_communication", default=0, choices=[0,1],type=int, help="whether to use communication based algorithm")
    # maddpg's setting
    parser.add_argument("--use_centralized_critic", default=1, choices=[0,1], type=int, help="whether to use only one centralized critic")
    # dpmac update
    parser.add_argument("--dpmac_version", default="v1", choices=["v1"], type=str, help="two versions of dpmac. both work but varies in speed (v1 faster than v2). V2 is discarded.")
    parser.add_argument("--use_state_msg_alignment", default=1, choices=[0, 1], type=int, help="whether to perform state-msg alignment")
    # communication's setting
    parser.add_argument("--sender_method", choices=['mlp', 'attention'], default='mlp')
    parser.add_argument("--receiver_method", choices=['mlp', 'attention'], default='mlp')
    parser.add_argument("--gaussian_receiver", choices=[0,1], default=1, type=int, help="whether to use stochastic receiver")
    parser.add_argument("--message_dim", default=5, type=int, help="Size of message dimension")
    parser.add_argument("--dpmac_privacy_priori", default=1, type=int, help="Whether to use the privacy priori for dpmac to adjust the learned message distribution")
    parser.add_argument("--sender_lr", default=1e-4, type=float)
    parser.add_argument("--receiver_lr", default=1e-4, type=float)
    parser.add_argument("--effectiness_loss",  default=False, action='store_true', help="whether to use message effectiveness loss")
    parser.add_argument("--protocol_grad_clip", default=True, action='store_false', help="whether perform grad clip on sender and receiver")
    parser.add_argument("--msg_include_id", default=0,type=int, help="whether message includes agent id explicity")
    # neural network
    parser.add_argument("--comm_hidden_dim", default=32, type=int)
    parser.add_argument("--atom", default=51, type=int)
    # private communication (some loss functions)
    parser.add_argument("--privacy_budget", default=0.5, type=float, help="the privacy budge epsilon") 
    parser.add_argument("--DP_delta", default=1e-4, type=float, help="the failure probability delta") 
    parser.add_argument("--noise_scale", default=1, type=float, help="the scale factor of the noise added in DP")
    # arguments for achieving theoretical dp
    parser.add_argument("--achieve_dp", default=0, type=int, help="whether to achieve dp theoretically in experiment")
    parser.add_argument("--achieve_dp_in_receiver", default=0, type=int, help="whether to achieve dp in receiver side")
    parser.add_argument("--achieve_dp_in_sender", default=1, type=int, help="whether to achieve dp in sender side")
    parser.add_argument("--delta", default=1e-4, type=float, help="delta in (epsilon, delta)-DP")
    parser.add_argument("--noise_method", default='gauss', choices=['gauss', 'laplace'], help='the noise way')
    parser.add_argument("--sensitivity", default=2.0, type=float, help="sensitivity in DP. should be the same as the clip value.")
    parser.add_argument("--episodic_dp", default=0, type=int, help="whether to achieve the episode-level (epsilon, delta)-DP. if true, the noise variance will be scaled up by episode_length")
    # wandb
    parser.add_argument('--use_wandb', default=0, type=int, choices=[0,1],
                        help="Whether to use weights&biases, if not, use tensorboardX instead")
    parser.add_argument("--wandb_project", default="debug", type=str, help="wandb project name")
    parser.add_argument("--wandb_name", default="0", type=str, help="wandb name")
    parser.add_argument("--wandb_group", default="debug", type=str, help="wandb group name")
    ##### end #####
    return parser
