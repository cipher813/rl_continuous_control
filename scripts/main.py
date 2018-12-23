# import os
# import re
# import gym
# import time
# import torch
# import pickle
# import random
# import datetime
# import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt
# %matplotlib inline

from agents.DDPG import DDPG
from agents.TD3 import TD3

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
# PATH = "/home/cipher813/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

# def pickle_results(RESULT_PATH, env_name, timestamp,pkl_file):
#     """ Save results to pickle file """
#     pklpath = RESULT_PATH + f"{env_name}_{timestamp}_ResultDict.pkl"
#     with open(pklpath, 'wb') as handle:
#         pickle.dump(pkl_file, handle)
#     print(f"Scores pickled at {pklpath}")
#
# def train_policy(RESULT_PATH, env_name, agent_dict, n_episodes=20000, max_t=700,
#                  learn_every=20, num_learn=10, score_threshold=300.0):
#     """Run policy train.
#
#     Arguments:
#     env_name (str): name of environment (ie for gym or unity)
#     agent_dict (dict): agents to train
#     n_episodes (int): max number of episodes to train
#     max_t (int): max timesteps per episode
#     learn_every (int): update network timestep increment
#     num_learn (int): number of times to update network per every timestep increment (ie learn_every)
#     score_threshold (float): once training reaches this average, break train
#     """
#     timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]
#     start = time.time()
#     result_dict = {}
#     env = gym.make(env_name)
#     env.seed(10)
#     for k,v in agent_dict.items():
#         policy = v(state_size=env.observation_space.shape[0],
#                   action_size=env.action_space.shape[0],
#                   max_action=float(env.action_space.high[0]),
#                   random_seed=10)
#
#         scores_deque = deque(maxlen=100)
#         scores = []
#         max_score = -np.Inf
#         for i_episode in range(1, n_episodes+1):
#             state = env.reset()
#             policy.reset()
#             score = 0
#             for t in range(max_t):
#                 action = policy.act(state)
#                 next_state, reward, done, _ = env.step(action)
#                 policy.step(state, action, reward, next_state, done)
#                 score += reward
#                 state = next_state
#
#                 if t%learn_every==0:
#                     for _ in range(num_learn):
#                         policy.start_learn()
#
#                 if done:
#                     break
#             scores_deque.append(score)
#             scores.append(score)
#             end = time.time()
#             print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tRuntime: {(end-start)/60:.1f}',end="")
#             if i_episode % 100 == 0 or np.average(scores_deque)>=score_threshold:
#                 fap = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_actor.pth'
#                 # fap = "../results/checkpoint_actor.pth"
#                 torch.save(policy.actor.state_dict(), fap)
#                 fcp = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_critic.pth'
#                 # fcp = "../results/checkpoint_critic.pth"
#                 torch.save(policy.critic.state_dict(), fcp)
#                 print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tRuntime: {(end-start)/60:.1f}')
#             if np.average(scores_deque)>score_threshold:
#                 break
#         end = time.time()
#         result_dict[k] = {
#                           "Scores": scores,
#                           "Runtime":np.round((end-start)/60,1)
#                           }
#     pickle_results(RESULT_PATH, env_name, timestamp, result_dict)
#     return scores

agent_dict = {
              "DDPG":DDPG,
              # "TD3":TD3,
             }

scores = train_policy(RESULT_PATH, 'BipedalWalker-v2',agent_dict)
