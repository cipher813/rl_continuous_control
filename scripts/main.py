from agents.DDPG import DDPG
from agents.TD3 import TD3
from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
# PATH = "/home/cipher813/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

agent_dict = {
              "DDPG":DDPG,
              # "TD3":TD3,
             }

# scores = train_gym(PATH, "BipedalWalker-v2",agent_dict)
scores = train_unity(PATH, "Reacher1.app",agent_dict)
