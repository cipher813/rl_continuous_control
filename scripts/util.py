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
# import matplotlib.pyplot as plt
# %matplotlib inline

# from agents.DDPG import DDPG
# from agents.TD3 import TD3

def pickle_results(RESULT_PATH, env_name, timestamp,pkl_file):
    """ Save results to pickle file """
    pklpath = RESULT_PATH + f"{env_name}_{timestamp}_ResultDict.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(pkl_file, handle)
    print(f"Scores pickled at {pklpath}")

def prep_gym(env_name, random_seed):
    env = gym.make(env_name)
    env.seed(random_seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    return env, state_size, action_size, max_action

def step_gym(env, action):
    next_state, reward, done, _ = env.step(action)
    return next_state, reward, done

def prep_unity(DATA_PATH):
    env = UnityEnvironment(file_name=DATA_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset()[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    max_action = float(action_size)
    return env, env_info, brain_name, brain, state_size, action_size, max_action

def step_unity(env, action):
    state = env_info.vector_observations[0]
    action = agent.act(state,e)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]
    return next_state, reward, done # state, action, env_info

def train_policy(RESULT_PATH, env_name, agent_dict, n_episodes=20000, max_t=700,
                 learn_every=20, num_learn=10, score_threshold=300.0, random_seed=10):
    """Run policy train.

    Arguments:
    env_name (str): name of environment (ie for gym or unity)
    agent_dict (dict): agents to train
    n_episodes (int): max number of episodes to train
    max_t (int): max timesteps per episode
    learn_every (int): update network timestep increment
    num_learn (int): number of times to update network per every timestep increment (ie learn_every)
    score_threshold (float): once training reaches this average, break train
    """
    timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]
    start = time.time()
    result_dict = {}
    env, state_size, action_size, max_action = prep_gym(env_name, random_seed) # gym
    for k,v in agent_dict.items():
        policy_name = k
        policy = v(state_size,action_size,max_action, random_seed)
        scores_deque = deque(maxlen=100)
        scores = []
        max_score = -np.Inf
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            policy.reset()
            score = 0
            for t in range(max_t):
                action = policy.act(state)
                # next_state, reward, done, _ = env.step(action)
                next_state, reward, done = step_gym(env, action) # gym
                policy.step(state, action, reward, next_state, done)
                score += reward
                state = next_state

                if t%learn_every==0:
                    for _ in range(num_learn):
                        policy.start_learn()

                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            end = time.time()
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tRuntime: {(end-start)/60:.1f}',end="")
            if i_episode % 100 == 0 or np.average(scores_deque)>=score_threshold:
                fap = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_actor.pth'
                # fap = "../results/checkpoint_actor.pth"
                torch.save(policy.actor.state_dict(), fap)
                fcp = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_critic.pth'
                # fcp = "../results/checkpoint_critic.pth"
                torch.save(policy.critic.state_dict(), fcp)
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}\tRuntime: {(end-start)/60:.1f}')
            if np.average(scores_deque)>score_threshold:
                break
        end = time.time()
        result_dict[k] = {
                          "Scores": scores,
                          "Runtime":np.round((end-start)/60,1)
                          }
    pickle_results(RESULT_PATH, env_name, timestamp, result_dict)
    return scores
