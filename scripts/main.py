"""
Deep Q Network (DQN) main train file.
Project 1: Navigation
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
December 2018
"""
# from util import train_gym, train_unity, train_envs
from util import train_envs
from agent import DDPG #, TD3

import os
import re
import datetime

# path information
PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/" # for mac
# PATH = "/home/bcm822_gmail_com/rl_continuous_control/" # for google cloud
CHART_PATH = PATH + "charts/"
CHECKPOINT_PATH = PATH + "models/"

timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

seed = 0
n_episodes=3000
max_t=1000
e_start=0.4
e_end=0.01
e_decay=0.995

rd = {}
env_dict = {
            # OpenAI Gym
            "BipedalWalker-v2":["gym",300.0],
            "Pendulum-v0":["gym",-250]

            # unity, for mac
           # "Reacher1.app":["unity",30.0],
            }

agent_dict = {
              "DDPG":DDPG,
              # "TD3":TD3,
             }

rd = train_envs(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, timestamp, env_dict, seed,
                n_episodes,max_t,e_start,e_end,e_decay)
