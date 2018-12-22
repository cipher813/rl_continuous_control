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

def train_policy(env_name, agent_dict, n_episodes=20000, max_t=700, score_threshold=300.0):
    result_dict = {}
    env = gym.make(env_name)
    env.seed(10)
    for k,v in agent_dict.items():
        agent = v(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.shape[0],
                  max_action=float(env.action_space.high[0]),
                  random_seed=10)

        scores_deque = deque(maxlen=100)
        scores = []
        max_score = -np.Inf
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            agent.reset()
            score = 0
            for t in range(max_t):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            end = time.time()
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tScore: {score:.2f}\tRuntime: {(end-start)/60:.1f}',end="")
            if i_episode % 100 == 0:
                torch.save(agent.actor.state_dict(), RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_actor.pth')
                torch.save(agent.critic.state_dict(), RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_critic.pth')
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
