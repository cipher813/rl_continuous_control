"""
BUG due to bug in built version of Unity, cannot load two unity environments in same loop
https://github.com/Unity-Technologies/ml-agents/issues/1167
This may be fixed in mlagents.envs but have not verified.
"""
from agents.DDPG import DDPG
from agents.D4PG import D4PG
from agents.DDPGplus import DDPGplus
# from agents.TD3 import TD3
from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
# PATH = "/home/cipher813/rl_continuous_control/"
# PATH = "/home/bcm822_gmail_com/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

env_dict = {
            # "Reacher20":["unity","Reacher20.app",0.0], # 30.0
            # "Reacher20":["unity","Reacher_Linux_NoVis2/Reacher.x86_64",30.0],
            "Reacher1":["unity","Reacher1.app",30.0], # 30.0
            # "Reacher1":["unity","Reacher_Linux_NoVis1/Reacher.x86_64",30.0],
            # "Pendulum":["gym","Pendulum-v0",-2000.0],
            # "BipedalWalker":["gym","BipedalWalker-v2",300.0] # 300.0
            }

agent_dict = {
              # "DDPG":DDPG,
              # "DDPGplus":DDPGplus,
              "D4PG":D4PG,
              # "TD3":TD3,
             }

results = train_envs(PATH, env_dict, agent_dict)
print(results)
