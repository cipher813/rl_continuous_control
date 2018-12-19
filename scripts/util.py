# Inspired by implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

import os
import gym
import time
import torch
import numpy as np

# from TD3 import *
# from DDPG import *
# from DDPG2 import *
from agents.TD3 import *
from agents.DDPG import *
from agents.DDPG2 import *
from network import *

def make_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Creating folder at {path}")

def pickle_results(RESULT_PATH, result_dict, timestamp, policy_name):
    pklpath = RESULT_PATH + f"{timestamp}-ResultDict-{agent_name}.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(result_dict, handle)
    print(f"Scores pickled at {pklpath}")

def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

def train_policy(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
                eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                policy_freq, RESULT_PATH, expl_noise):
    start = time.time()
    result_dict = {}
    for k,v in policy_dict.items():
        policy_name = k
        filename = f"{policy_name}-{env_name}-{str(seed)}"
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
        evaluations = [evaluate_policy(env, policy)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        while total_timesteps < max_timesteps: # 1 million
            if done: # only executes if done is True
                if total_timesteps !=0:
                    print((f"E: {episode_num} R: {episode_reward:.2f} Step, Ep: {episode_timesteps} Tot: {total_timesteps}"))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                if timesteps_since_eval > eval_freq: # 5000
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(env, policy))

                    policy.save(filename, directory=RESULT_PATH)
                    np.save(RESULT_PATH + filename,evaluations)

                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                if episode_num >0:
                    scores.append([reward])
                episode_num +=1

            if total_timesteps < start_timesteps: #10,000
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(obs))
                if expl_noise != 0:
                    action = ((action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).
                               clip(env.action_space.low, env.action_space.high))

            new_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_reward += reward

            replay_buffer.add((obs, new_obs, action, reward, done_bool))
            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        evaluations.append(evaluate_policy(env, policy))
        policy.save(filename,directory=RESULT_PATH)
        np.save(RESULT_PATH + filename,evaluations)
        print(f"Model saved as {filename}")
        end = time.time()
        result_dict[policy_name] = {
                        "scores":scores,
                        "evaluations": evaluations,
                        "clocktime":round((end-start)/60,2)
                        }
    pickle_results(RESULT_PATH, result_dict, timestamp, policy_name)
    return result_dict

def train_envs(RESULT_PATH, MODEL_PATH, policy_dict, timestamp, env_dict, seed,
               start_timesteps, max_timesteps,eval_freq, batch_size, discount,
               tau, policy_noise, noise_clip,policy_freq, expl_noise):
    """Main train function for all envs in env_dict."""
    rd = {}
    for k,v in env_dict.items():
        start = time.time()
        env_name = k
        platform = v[0]
        print(f"Begin training {env_name}-{platform}.")
        score_target = v[1]
        print(f"Module: {env_name}-{platform}")
        results = train_policy(timestamp, env_name, seed, policy_dict, start_timesteps, max_timesteps,
                               eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                               policy_freq, RESULT_PATH, expl_noise)
        rd[env_name] = results
        end = time.time()
        print(f"Finished training {env_name}-{platform} in {(end-start)/60:.2f} minutes.")
    pklpath = CHART_PATH + f"{timestamp}-ResultDict-All.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(rd, handle)
        print(f"Scores pickled at {pklpath}")
    return rd
