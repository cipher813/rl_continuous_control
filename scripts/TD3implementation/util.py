# Inspired by implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

import os
import gym
import time
import torch
import pickle
import numpy as np

from agents.TD3 import *
from agents.DDPG import *
from agents.DDPG2 import *
from network import *

def make_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Creating folder at {path}")

def pickle_results(RESULT_PATH, env_name, result_dict, timestamp, policy_name):
    pklpath = RESULT_PATH + f"{env_name}_{timestamp}_ResultDict_{policy_name}.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(result_dict, handle)
    print(f"Scores pickled at {pklpath}")

def evaluate_policy_gym(env, policy, eval_episodes=100):
    """ Runs policy for X episodes and returns average reward """
    avg_reward = 0.
    for i in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            # action = policy.select_action(torch.from_numpy(np.array(state)))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print(f"Evaluation in {eval_episodes} episodes ({i} steps): {avg_reward:.2f}")
    return avg_reward

def evaluate_policy_unity(env, policy, eval_episodes=10):
    """ Runs policy for X episodes and returns average reward """
    avg_reward = 0.
    for i in range(eval_episodes):
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = policy.select_action(np.array(states))
            actions = np.clip(actions,-1,1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            # action = policy.select_action(torch.from_numpy(np.array(state)))
            # state, reward, done, _ = env.step(action)
            avg_reward += scores
            if np.any(dones):
                break
        print(f"Total score (averaged over agents) this episode: {np.mean(scores)}")
    avg_reward /= eval_episodes
    # print(f"Evaluation in {eval_episodes} episodes ({str(i)} steps): {avg_reward:.2f}")
    return avg_reward

def train_gym(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
                eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                policy_freq, DATA_PATH, RESULT_PATH, expl_noise, score_target, platform):
    start = time.time()
    result_dict = {}
    for k,v in policy_dict.items():
        policy_name = k
        filename = f"{env_name}_{timestamp}_{policy_name}_{str(seed)}"
        print(f"**Running {filename}**")

        env = gym.make(env_name)
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        policy = v(state_dim, action_dim, max_action)
        scores = []
        replay_buffer = ReplayBuffer()
        evaluations = [evaluate_policy_gym(env, policy)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        while total_timesteps < max_timesteps: # less than 1 million timesteps

            # take average reward of last 100 episodes
            avg_score = np.mean(scores[-100:]) if len(scores)>100 else np.mean(scores) if len(scores)>0 else 0

            # if average score is greater than score target, break while loop
            if avg_score>=score_target:
                break
            if done: # only executes if done is True
                if total_timesteps !=0:
                    print((f"E: {episode_num} R: {episode_reward:.1f} Step, Ep: {episode_timesteps} Tot: {total_timesteps}"))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                if timesteps_since_eval > eval_freq: # 5000
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy_gym(env, policy))

                    policy.save(filename, directory=RESULT_PATH)
                    np.save(RESULT_PATH + filename,evaluations)

                state = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num +=1

            if total_timesteps < start_timesteps: #10,000
                # select action randomly
                action = env.action_space.sample() # gym
            else:
                # select action according to policy
                action = policy.select_action(np.array(state))
                # if expl_noise != 0:
                action = ((action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).
                           clip(env.action_space.low, env.action_space.high))
            next_state, reward, done, _ = env.step(action) # gym
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done) # gym
            episode_reward += reward
            scores.append([reward])

            # store data in replay buffer
            replay_buffer.add((state, next_state, action, reward, done_bool))
            state = next_state

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # final evaluation
        evaluations.append(evaluate_policy_gym(env, policy))
        policy.save(filename,directory=RESULT_PATH)
        np.save(RESULT_PATH + filename,evaluations)
        print(f"Model saved as {filename}")
        end = time.time()
        result_dict[policy_name] = {
                        "scores":scores,
                        "evaluations": evaluations,
                        "clocktime":round((end-start)/60,2)
                        }
    pickle_results(RESULT_PATH, env_name, result_dict, timestamp, policy_name)
    return result_dict

def train_unity(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
                eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                policy_freq, DATA_PATH, RESULT_PATH, expl_noise, score_target, platform):
    from unityagents import UnityEnvironment
    start = time.time()
    result_dict = {}

    for k,v in policy_dict.items():
        policy_name = k
        filename = f"{env_name}_{timestamp}_{policy_name}_{str(seed)}"
        print(f"**Running {filename}**")

        file_name = DATA_PATH + env_name
        print(f"Opening agent from {file_name}")
        env = UnityEnvironment(file_name=file_name)
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]

        num_agents = len(env_info.agents)
        print(f"Number of agents: {num_agents}")

        # env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = env_info.vector_observations.shape[1]
        action_dim = brain.vector_action_space_size
        print(f"Dims, State: {state_dim} Action: {action_dim}")
        max_action = float(action_dim)
        print(f"Max Action: {max_action}")
        policy = v(state_dim, action_dim, max_action)
        scores = []
        replay_buffer = ReplayBuffer()
        evaluations = [evaluate_policy_unity(env, policy)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        while total_timesteps < max_timesteps: # less than 1 million timesteps

            # take average reward of last 100 episodes
            avg_score = np.mean(scores[-100:]) if len(scores)>100 else np.mean(scores) if len(scores)>0 else 0

            # if average score is greater than score target, break while loop
            if avg_score>=score_target:
                break
            if done: # only executes if done is True
                if total_timesteps !=0:
                    print((f"E: {episode_num} R: {episode_reward:.1f} Step, Ep: {episode_timesteps} Tot: {total_timesteps}"))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                if timesteps_since_eval > eval_freq: # 5000
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy_unity(env, policy))

                    policy.save(filename, directory=RESULT_PATH)
                    np.save(RESULT_PATH + filename,evaluations)

                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num +=1

            state = env_info.vector_observations[0]
            action = policy.select_action(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            # env.step(state, action, reward, next_state, done)

            episode_reward += reward
            scores.append([reward])

            # store data in replay buffer
            replay_buffer.add((state, next_state, action, reward, done))
            # state = new_obs
            state = next_state

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # final evaluation
        evaluations.append(evaluate_policy_unity(env, policy))
        policy.save(filename,directory=RESULT_PATH)
        np.save(RESULT_PATH + filename,evaluations)
        print(f"Model saved as {filename}")
        end = time.time()
        result_dict[policy_name] = {
                        "scores":scores,
                        "evaluations": evaluations,
                        "clocktime":round((end-start)/60,2)
                        }
    pickle_results(RESULT_PATH, env_name, result_dict, timestamp, policy_name)
    return result_dict

def train_envs(DATA_PATH, RESULT_PATH, MODEL_PATH, policy_dict, timestamp, env_dict, seed,
               start_timesteps, max_timesteps,eval_freq, batch_size, discount,
               tau, policy_noise, noise_clip,policy_freq, expl_noise):
    """Main train function for all envs in env_dict."""
    rd = {}
    for k,v in env_dict.items():
        start = time.time()
        env_name = k
        platform = v[0]
        score_target = v[1]
        if platform=="gym":
            results = train_gym(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
                                   eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                                   policy_freq, DATA_PATH, RESULT_PATH, expl_noise, score_target, platform)
        else:
            results = train_unity(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
                                  eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                                  policy_freq, DATA_PATH, RESULT_PATH, expl_noise, score_target, platform)
        rd[env_name] = results
        end = time.time()
        print(f"Finished training {env_name}-{platform} in {(end-start)/60:.2f} minutes.")
    pklpath = RESULT_PATH + f"{timestamp}-ResultDict-All.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(rd, handle)
        print(f"Scores pickled at {pklpath}")
    return rd
