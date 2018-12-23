import os
import re
import gym
import time
import torch
import pickle
import random
import datetime
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline

from agents.DDPG import DDPG
from agents.TD3 import TD3

#PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_continuous_control/"
PATH = "/home/cipher813/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]
start = time.time()

# env_name = 'BipedalWalker-v2'
# env = gym.make(env_name)
# env.seed(10)
# agent = DDPG(state_size=env.observation_space.shape[0],
#              action_size=env.action_space.shape[0],
#              max_action=float(env.action_space.high[0]),
#              random_seed=10)

def pickle_results(RESULT_PATH, env_name, timestamp,pkl_file):
    pklpath = RESULT_PATH + f"{env_name}_{timestamp}_ResultDict.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(pkl_file, handle)
    print(f"Scores pickled at {pklpath}")

# def eval_episode(env, policy):
#     state = env.reset()
#     total_rewards = 0
#     while True:
#         action = policy.select_action(state)
#         state, reward, done, _ = env.step(action)
#         total_rewards += reward
#         if done:
#             break
#     return total_rewards
#
# def eval_episodes(env, policy, eval_episodes=20):
#     rewards = []
#     for ep in range(eval_episodes):
#         rewards.append(eval_episode(env, policy))
#     return rewards

def train_policy(env_name, agent_dict, n_episodes=20000, max_t=700, learn_every=20, score_threshold=300.0):
    result_dict = {}
    env = gym.make(env_name)
    env.seed(10)
    for k,v in agent_dict.items():
        policy = v(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0],
                  max_action=float(env.action_space.high[0]),
                  random_seed=10)

        scores_deque = deque(maxlen=100)
        scores = []
        max_score = -np.Inf
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            policy.reset()
            score = 0
            for t in range(max_t):
            # score_list = eval_episodes(env,policy)
                action = policy.select_action(state)
                next_state, reward, done, _ = env.step(action)
                policy.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if t%learn_every==0:
                    policy.start_learn()

                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            end = time.time()
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tRuntime: {(end-start)/60:.1f}',end="")
            if i_episode % 100 == 0 or np.average(scores_deque)>=score_threshold:
                # fap = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_actor.pth'
                fap = "../results/checkpoint_actor.pth"
                torch.save(policy.actor.state_dict(), fap)
                # fcp = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_critic.pth'
                fcp = "../results/checkpoint_critic.pth"
                torch.save(policy.critic.state_dict(), fcp)
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tRuntime: {(end-start)/60:.1f}')
            if np.average(scores_deque)>score_threshold:
                break
        result_dict[k] = scores
    pickle_results(RESULT_PATH, env_name, timestamp, result_dict)
    return scores

agent_dict = {
              "DDPG":DDPG,
              # "TD3":TD3,
             }

scores = train_policy('BipedalWalker-v2',agent_dict, score_threshold=300.0)
