import os
import re
import gc
import gym
import time
import torch
import pickle
import random
import datetime
import numpy as np
from collections import deque

def pickle_results(pklpath,pkl_file):
    """ Save results to pickle file """
    with open(pklpath, 'wb') as handle:
        pickle.dump(pkl_file, handle)
    print(f"Scores pickled at {pklpath}")

def calc_runtime(seconds):
    h = int(seconds//(60*60))
    seconds = seconds - h*60*60
    m = int(seconds//60)
    s = int(round(seconds - m*60,0))
    return "{:02d}:{:02d}:{:02d}".format(h,m,s)

def train_unity_ddpg(PATH, env_name, platform, env_path, policy, score_threshold,timestamp,start, n_episodes, max_t, num_agents):
    total_scores = []
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
    policy = policy(state_size,action_size,num_agents)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        policy.reset()
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
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores[-score_length:])
        end = time.time()
        print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}',end=" ")
        if i_episode % 20 == 0 or total_average_score>=score_threshold:
            fap = PATH + f'results/{env_name}_{timestamp}_checkpoint_actor.pth'
            torch.save(policy.actor.state_dict(), fap)
            fcp = PATH + f'results/{env_name}_{timestamp}_checkpoint_critic.pth'
            torch.save(policy.critic.state_dict(), fcp)
            print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}')
        if total_average_score>score_threshold:
            print(f"Solved in {i_episode} and {calc_runtime(end-start)}")
            break
    # env.reset()
    env.close()
    # seconds = 30
    # print(f"Sleeping for {seconds} seconds to close the damn unity environment...")
    # time.sleep(seconds)
    return total_scores

def train_gym_ddpg(PATH, env_name, platform, env_path, policy, score_threshold,timestamp,start, n_episodes, max_t,num_agents):
    total_scores = []
    import gym
    env = gym.make(env_path)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    policy = policy(state_size,action_size,num_agents)
    for i_episode in range(1, n_episodes+1):
        states = env.reset()
        scores = np.zeros(num_agents)
        policy.reset()
        for t in range(max_t):
            actions = policy.act(states)
            next_states, rewards, dones, _ = env.step(actions)
            policy.step(states, actions, rewards, next_states, dones, t)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score_length = len(total_scores) if len(total_scores)<100 else 100
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores[-score_length:])
        end = time.time()
        print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}',end=" ")
        if i_episode % 20 == 0 or total_average_score>=score_threshold:
            fap = PATH + f'results/{env_name}_{timestamp}_checkpoint_actor.pth'
            torch.save(policy.actor.state_dict(), fap)
            fcp = PATH + f'results/{env_name}_{timestamp}_checkpoint_critic.pth'
            torch.save(policy.critic.state_dict(), fcp)
            print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}')
        if total_average_score>score_threshold:
            print(f"Solved in {i_episode} and {calc_runtime(end-start)}")
            break
    return total_scores

def train_ddpg(PATH, env_name, platform, env_path, policy_name, policy, score_threshold,
                 timestamp,train_mode,n_episodes=10000, max_t=1000, num_agents=1):
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
    start = time.time()
    # RESULT_PATH = PATH + "results/"
    result_dict = {}
    # finished = False
    # i_episode = 0
    # while finished == False and i_episode < n_episodes:
    if platform=="unity":
        total_scores = train_unity_ddpg(PATH, env_name, platform, env_path, policy, score_threshold,timestamp,start, n_episodes, max_t, num_agents)
    elif platform=="gym":
        total_scores = train_gym_ddpg(PATH, env_name, platform, env_path, policy, score_threshold,timestamp,start, n_episodes, max_t, num_agents)
    else:
        print("Platform must be either 'unity' or 'gym'.")
        # break
    # end = time.time()
    # result_dict[(env_name, policy_name)] = {
    #                   "Scores": total_scores,
    #                   "Runtime":calc_runtime(end-start)
    #                   }
    # print(f"Updated Result Dictionary:\n{result_dict}")
    # pklpath = PATH + f"results/{timestamp}_{env_name}_ResultDict.pkl"
    # pickle_results(pklpath, result_dict)
    # # result_dict[(env_name,agent_name)] = scores
    # gc.collect()
    return result_dict

def train_envs(PATH, env_dict, agent_dict, train_mode):
    timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]
    result_dict = {}
    for k,v in env_dict.items():
        if v[2]==train_mode:
            env_name = k
            platform = v[0]
            env_path = v[1]
            score_threshold = v[3]
            print(f"Train Mode: {v[2].title()}-Agent")
            for k, v in agent_dict.items():
                start = time.time()
                policy_name = k
                print(f"Environment: {env_name}\tPolicy: {policy_name}")
                policy = v[0]
                mode = v[1] # single, multi or both
                if mode==train_mode or mode=="both":
                    if "DDPG" in policy_name or "D4PG" in policy_name:
                        total_scores = train_ddpg(PATH, env_name, platform, env_path, policy_name, policy, score_threshold, timestamp,train_mode)
                    end = time.time()
                    result_dict[(env_name, policy_name)] = {
                                      "Scores": total_scores,
                                      "Runtime":calc_runtime(end-start)
                                      }
                    print(f"Updated Result Dictionary:\n{result_dict}")
                    pklpath = PATH + f"results/{timestamp}_{env_name}_ResultDict.pkl"
                    pickle_results(pklpath, result_dict)
                    # result_dict[(env_name,agent_name)] = scores
                    gc.collect()
    return result_dict
