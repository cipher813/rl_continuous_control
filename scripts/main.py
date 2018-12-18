# Inspired by implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

import numpy as np
import torch
import gym
import os
from agent import TD3
from util import train_agent

env_name = "BipedalWalker-v2"
seed = 0
directory = "./pytorch_models"
start_timesteps = 1e4
eval_freq = 5e3
max_timesteps = 1e6
save_models = True
expl_noise = 0.1
batch_size = 100
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

policy_dict = {"TD3":TD3}

train_agent(env_name, seed, policy_dict, start_timesteps, max_timesteps,
            eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
            policy_freq, directory, save_models, expl_noise)
