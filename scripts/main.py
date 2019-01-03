"""
Project 2: Continuous Control
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
January 2019

BUG due to bug in built version of Unity, cannot load two unity environments in same loop
https://github.com/Unity-Technologies/ml-agents/issues/1167
This may be fixed in mlagents.envs but have not verified.
"""
from agents.DDPG import DDPG
# from agents.D4PG import D4PG
# from agents.DDPGplus import DDPGplus
# from agents.TD3 import TD3
from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
# PATH = "/home/cipher813/rl_continuous_control/"
# PATH = "/home/bcm822_gmail_com/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

# Number of agents to train.  Must be "single" or "multi."
# BUG cannot run more than one unity environment in same loop (so cannot be "both")
TRAIN_MODE = "multi"

env_dict = {
            "Reacher20":["unity","Reacher20.app","multi",30.0], # 30.0
            # "Reacher20":["unity","Reacher_Linux_NoVis2/Reacher.x86_64","multi",30.0],
            "Reacher1":["unity","Reacher1.app","single",30.0], # 30.0
            # "Reacher1":["unity","Reacher_Linux_NoVis1/Reacher.x86_64","single",30.0],
            "Pendulum":["gym","Pendulum-v0","single",2000.0],
            "BipedalWalker":["gym","BipedalWalker-v2","single",300.0] # 300.0
            }

# BUG cannot run more than one agent in a single unity environment in same loop
# per open issue at https://github.com/Unity-Technologies/ml-agents/issues/1167
agent_dict = {
              "DDPG":[DDPG,"both"],
              # "DDPGplus":[DDPGplus,"both"],
              # "D4PG":[D4PG,"single"],
              # "TD3":[TD3,"both"]
             }

results = train_envs(PATH, env_dict, agent_dict,TRAIN_MODE)
print(results)
