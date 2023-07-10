#!/bin/bash

use_wandb=1

seed=${1}
algo=${2} # rdpmaddpg
scenario=${3}
privacy_budget=${4}
achieve_dp=${5}
achieve_dp_in_sender=${6}
date=${7}
gpu_used=${8}
episodic_dp=${9} # whether to achieve (\epsilon,\delta)-DP on episode-level, setting 1 for fig 6 only

achieve_dp_in_receiver=0

env="MPE"

if [ ${scenario} = "po_spread" ]; then
    num_agents=3
elif [ ${scenario} = "po_reference" ]; then
    num_agents=2
elif [ ${scenario} = "simple_spread" ]; then
    num_agents=3
elif [ ${scenario} = "simple_adversary" ]; then
    num_agents=3
elif [ ${scenario} = "simple_crypto" ]; then
    num_agents=3
else
    echo "scenario is not defined"
    exit
fi

if [ ${algo} = "rdpmaddpg" ]; then
    use_communication=1
    sender="mlp"
    receiver="attention"  
else
    echo "algo is not defined"
    exit
fi

sensitivity=0.5
privacy_sample_num=5
po_spread_view_radius=1.2
use_reward_normalization=1
use_centralized_critic=0
use_state_msg_alignment=0
### end ###
### fixed param ###
num_landmarks=3
batch_size=128
sender_lr=7e-4
receiver_lr=7e-4
message_dim=8
comm_hidden_dim=32
hidden_size=128
msg_include_id=0
num_env_steps=1000000
buffer_size=10000
### end ###
exp="${algo}_eps${privacy_budget}_dp${achieve_dp}_dps${achieve_dp_in_sender}_EDP${episodic_dp}_${date}"
wandb_group=${exp}

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=${gpu_used} nohup python src/train_mpe.py --env_name ${env} \
                                        --algorithm_name ${algo} \
                                        --experiment_name ${exp} \
                                        --scenario_name ${scenario} \
                                        --num_agents ${num_agents} \
                                        --num_landmarks ${num_landmarks} \
                                        --seed ${seed} \
                                        --episode_length 25 \
                                        --actor_train_interval_step 1 \
                                        --tau 0.005 \
                                        --lr 7e-4 \
                                        --sender_lr ${sender_lr} \
                                        --receiver_lr ${receiver_lr} \
                                        --use_rnn_layer \
                                        --use_naive_recurrent_policy \
                                        --num_env_steps ${num_env_steps} \
                                        --use_reward_normalization ${use_reward_normalization} \
                                        --hidden_size ${hidden_size} \
                                        --batch_size ${batch_size} \
                                        --use_communication ${use_communication} \
                                        --receiver_method ${receiver} \
                                        --sender_method ${sender} \
                                        --comm_hidden_dim ${comm_hidden_dim} \
                                        --message_dim ${message_dim} \
                                        --msg_include_id ${msg_include_id} \
                                        --buffer_size ${buffer_size} \
                                        --use_wandb ${use_wandb}\
                                        --achieve_dp ${achieve_dp} \
                                        --po_spread_view_radius ${po_spread_view_radius} \
                                        --use_state_msg_alignment ${use_state_msg_alignment} \
                                        --privacy_sample_num ${privacy_sample_num} \
                                        --privacy_budget ${privacy_budget} \
                                        --sensitivity ${sensitivity} \
                                        --achieve_dp_in_sender ${achieve_dp_in_sender} \
                                        --achieve_dp_in_receiver ${achieve_dp_in_receiver} \
                                        --wandb_group ${wandb_group} \
                                        --wandb_project "[Release]"${scenario} \
                                        --episodic_dp ${episodic_dp} &
                                    
echo "training is done!"

