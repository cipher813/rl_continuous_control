"""
BUG due to bug in built version of Unity, cannot load two unity environments in same loop
https://github.com/Unity-Technologies/ml-agents/issues/1167
This may be fixed in mlagents.envs but have not verified.
"""


from agents.DDPG import DDPG
# from agents.TD3 import TD3
from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
# PATH = "/home/cipher813/rl_continuous_control/"
# PATH = "/home/bcm822_gmail_com/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

env_dict = {
            "Reacher20":["unity","Reacher20.app",30.0], # 30.0
            # "Reacher1":["unity","Reacher1.app",0.01],
            # "BipedalWalker-v2":["gym","BipedalWalker-v2",0.0] # 300.0
            }

agent_dict = {
              "DDPG":DDPG,
              # "TD3":TD3,
             }

# scores = train_gym(PATH, "BipedalWalker-v2",agent_dict)
# scores = train_unity(PATH, "Reacher20","Reacher20.app",agent_dict)
# scores = train_unity(PATH,"Reacher20","Reacher_Linux_NoVis2/Reacher.x86_64",agent_dict)

# def train_envs(PATH, env_dict, agent_dict):
#     result_dict = {}
#     for k,v in env_dict.items():
#         env_name = k
#         platform = v[0]
#         env_path = v[1]
#         if platform=="unity":
#             scores = train_unity(PATH, env_name, env_path, agent_dict)
#         else:
#             scores = train_gym()
#         result_dict[env_name] = scores
#     return result_dict

results = train_envs(PATH, env_dict, agent_dict)
