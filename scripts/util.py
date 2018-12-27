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

def pickle_results(RESULT_PATH, env_name, timestamp,pkl_file):
    """ Save results to pickle file """
    pklpath = RESULT_PATH + f"{env_name}_{timestamp}_ResultDict.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(pkl_file, handle)
    print(f"Scores pickled at {pklpath}")

def calc_runtime(seconds):
    h = int(seconds//(60*60))
    seconds = seconds - h*60*60
    m = int(seconds//60)
    s = seconds - m*60
    return "{:02d}:{:02d}:{:.0f}".format(h,m,s)

def step_unity(env, action):
    state = env_info.vector_observations[0]
    action = agent.act(state,e)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]
    return next_state, reward, done # state, action, env_info

def train_unity(PATH, env_name, env_path, agent_dict, n_episodes=20000, max_t=1000,
                 score_threshold=30.0, learn_every=20, num_learn=10, random_seed=7):
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
    RESULT_PATH = PATH + "results/"
    timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]
    result_dict = {}

    from unityagents import UnityEnvironment
    env_path = PATH + f"data/{env_path}"
    env = UnityEnvironment(file_name=env_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print(f"Number of agents: {num_agents}")
    states = env_info.vector_observations
    state_size = states.shape[1]
    print(f"There are {states.shape[0]} agents.  Each observes a state with length {state_size}")
    print(f"The state for the first agent looks like:\n{states[0]}")
    action_size = brain.vector_action_space_size
    print(f"Size of each action: {action_size}")
    # max_action = float(action_size)

    for k,v in agent_dict.items():
        start = time.time()
        policy_name = k
        policy = v(state_size,action_size,random_seed) #max_action,num_agents,learn_freq,
        # total_scores_deque = deque(maxlen=100)
        total_scores = []
        # max_score = -np.Inf
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            scores = np.zeros(num_agents)
            policy.reset()
            # while True:
            for t in range(max_t):
                actions = policy.act(states)
                env_info = env.step(actions)[brain_name]        # send the action to the environment
                next_states = env_info.vector_observations
                rewards = env_info.rewards                   # get the reward
                dones = env_info.local_done
                policy.step(states, actions, rewards, next_states, dones, t)
                states = next_states
                scores += env_info.rewards

                if np.any(dones):
                    break
            score_length = len(total_scores) if len(total_scores)<100 else 100
            mean_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            # total_scores_deque.append(mean_score)
            total_scores.append(mean_score)
            total_average_score = np.mean(total_scores[-score_length:])
            end = time.time()
            print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}',end=" ")
            if i_episode % 20 == 0 or total_average_score>=score_threshold:
                fap = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_actor.pth'
                torch.save(policy.actor.state_dict(), fap)
                fcp = RESULT_PATH + f'{env_name}_{timestamp}_checkpoint_critic.pth'
                torch.save(policy.critic.state_dict(), fcp)
                print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}')
            if total_average_score>score_threshold:
                print(f"Solved in {i_episode} and {calc_runtime(end-start)}")
                break
        end = time.time()
        result_dict[k] = {
                          "Scores": total_scores,
                          "Runtime":calc_runtime(end-start)
                          }
    pickle_results(RESULT_PATH, env_name, timestamp, result_dict)
    return scores
