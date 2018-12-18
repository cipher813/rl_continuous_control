# Inspired by implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

import numpy as np
import torch
import gym
import os
# from TD3 import TD3
# from DDPG import DDPG
# from DDPG2 import DDPG2
# from util import make_paths, train_policy
from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
DATA_PATH = PATH + "data/"
MODEL_PATH = PATH + "models/"
RESULT_PATH = PATH + "results/"

make_paths([MODEL_PATH, RESULT_PATH])

env_name = "BipedalWalker-v2"
seed = 0
start_timesteps = 1e4
eval_freq = 5e3
max_timesteps = 1e6
batch_size = 100
discount = 0.99
tau = 0.005
expl_noise = 0.1
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

policy_dict = {
                "DDPG2":DDPG2,
                "DDPG":DDPG,
                "TD3":TD3
                }

train_policy(env_name, seed, policy_dict, start_timesteps, max_timesteps,
            eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
            policy_freq, RESULT_PATH, expl_noise)
