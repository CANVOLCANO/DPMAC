#!/bin/bash

seed=${1}
algo=${2} # rdpmaddpg
privacy_budget=${3}
achieve_dp=${4}
achieve_dp_in_sender=${5}
date=${6}
gpu_used=${7}
episodic_dp=${8} # whether to achieve (\epsilon,\delta)-DP on episode-level, setting 1 for fig 6 only

use_wandb=1
achieve_dp_in_receiver=0

env="I2C_PP"
scenario="pp"

if [ ${algo} = "rdpmaddpg" ]; then
    use_communication=1
    sender="mlp"
    receiver="attention"  
else
    echo "algo is not defined"
    exit
fi

echo "receiver: ${receiver}"
num_landmarks=3
batch_size=256
seed_max=1
num_env_steps=1000000
hidden_size=128
buffer_size=10000
use_reward_normalization=1
use_state_msg_alignment=0
use_centralized_critic=0
lr=7e-4
sender_lr=7e-4
receiver_lr=7e-4
message_dim=8
comm_hidden_dim=64
msg_include_id=0
gamma=0.95
num_agents=3
num_preys=2
sensitivity=0.5

exp="${algo}_eps${privacy_budget}_dp${achieve_dp}_dps${achieve_dp_in_sender}_EDP${episodic_dp}_${date}"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

wandb_group=${exp}
CUDA_VISIBLE_DEVICES=${gpu_used} nohup python src/train_pp.py --env_name ${env} \
                                        --algorithm_name ${algo} \
                                        --experiment_name ${exp} \
                                        --scenario_name ${scenario} \
                                        --num_agents ${num_agents} \
                                        --num_landmarks ${num_landmarks} \
                                        --seed ${seed} \
                                        --episode_length 40 \
                                        --actor_train_interval_step 1 \
                                        --tau 0.005 \
                                        --lr ${lr} \
                                        --sender_lr ${sender_lr} \
                                        --receiver_lr ${receiver_lr} \
                                        --use_rnn_layer \
                                        --use_naive_recurrent_policy \
                                        --num_env_steps ${num_env_steps} \
                                        --use_reward_normalization ${use_reward_normalization} \
                                        --batch_size ${batch_size} \
                                        --hidden_size ${hidden_size} \
                                        --use_communication ${use_communication} \
                                        --receiver_method ${receiver} \
                                        --comm_hidden_dim ${comm_hidden_dim} \
                                        --message_dim ${message_dim} \
                                        --msg_include_id ${msg_include_id} \
                                        --gamma 0.95 \
                                        --num_agents ${num_agents}\
                                        --num_preys ${num_preys}\
                                        --achieve_dp ${achieve_dp} \
                                        --buffer_size ${buffer_size} \
                                        --use_wandb ${use_wandb} \
                                        --wandb_group ${wandb_group} \
                                        --wandb_project "[Release]"${scenario} \
                                        --achieve_dp_in_sender ${achieve_dp_in_sender} \
                                        --achieve_dp_in_receiver ${achieve_dp_in_receiver} \
                                        --privacy_budget ${privacy_budget} \
                                        --sensitivity ${sensitivity} \
                                        --sender_method ${sender} \
                                        --episodic_dp ${episodic_dp} &
echo "training is done!"

