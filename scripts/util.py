# Inspired by implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

import gym
import torch
import numpy as np

from agent import *
from network import *

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
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
    return avg_reward

def train_agent(env_name, seed, policy_dict, start_timesteps, max_timesteps,
                eval_freq, batch_size, discount, tau, policy_noise, noise_clip,
                policy_freq, directory, save_models, expl_noise):
    for k,v in policy_dict.items():
        policy_name = k
        filename = f"{policy_name}-{env_name}-{str(seed)}"
        env = gym.make(env_name)
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        policy = v(state_dim, action_dim, max_action)

        replay_buffer = ReplayBuffer()

        evaluations = [evaluate_policy(env, policy)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True

        while total_timesteps < max_timesteps:
            if done:
                if total_timesteps !=0:
                    print(f"E: {episode_num} R: {episode_reward} Step,  Ep: {episode_timesteps} Tot: {total_timesteps}")
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                if timesteps_since_eval > eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(env, policy))

                    if save_models: policy.save(filename, directory=directory)
                    np.save(f"./results/{filename}",evaluations)

                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num +=1

            if total_timesteps < start_timesteps:
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
        if save_models: policy.save(f"{filename}",directory=directory)
        np.save(f"./results/{filename}",evaluations)
        print(f"Model saved as {filename}")
