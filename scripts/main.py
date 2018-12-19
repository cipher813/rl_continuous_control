# Inspired by implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
import os
import re
import gym
import datetime
import numpy as np

import torch

from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
# PATH = "/home/bcm822_gmail_com/rl_continuous_control/"
DATA_PATH = PATH + "data/"
MODEL_PATH = PATH + "models/"
RESULT_PATH = PATH + "results/"

make_paths([MODEL_PATH, RESULT_PATH])

timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

# env_name = "BipedalWalker-v2"
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

env_dict = {
            "BipedalWalker-v2":["gym",300.0],
            "LunarLanderContinuous-v2":["gym",200.0],
            "Pendulum-v0":["gym",-250.0]
            }

policy_dict = {
                "TD3":TD3,
                "DDPG2":DDPG2,
                "DDPG":DDPG
                }

# train_policy(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
#             eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
#             policy_freq, RESULT_PATH, expl_noise)

rd = train_envs(RESULT_PATH, MODEL_PATH, policy_dict, timestamp, env_dict, seed,
                start_timesteps, max_timesteps,eval_freq, batch_size, discount,
                tau, policy_noise, noise_clip,policy_freq, expl_noise)
