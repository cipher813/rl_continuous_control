<a name="report"></a>
# Reinforcement Learning Continuous Control: Project Report

## Report Table of Contents

[RL Environment](#environment)

[Algorithm](#algorithm)

[Hyperparameters](#hyperparameters)

[Network Architecture](#network)

[Next Steps](#nextsteps)

<a name="environment"></a>
## The Reinforcement Learning (RL) Environment

For [Project 2](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with teaching an agent to maintain a target position using the "Reacher" environment configured by Udacity on [Unity's ML-Agents platform](https://github.com/Unity-Technologies/ml-agents).  

In the environment, a double-jointed arm moves to target locations, with a reward of 0.1 provided for each step that the agent's hand is in the goal location.  The goal of the agent is to maintain its position at the target location for as many steps as possible.  

The state space consists of 33 variables corresponding to position, rotation, velocity and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector is between -1 and 1.  

Two Reacher environments are provided, with one consisting of a single agent and another with 20 identical agents.  For each environment, the agent(s) must achieve an average score of 30.0 over 100 episodes.  The project submission need only solve one of the two versions of the environment - but the DDPG implementation can actually solve both Reacher environments.  

For further information, see Udacity's [project github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

<a name="algorithm"></a>
## The Algorithm

In this project, we explored a variety of policies to solve this continuous state space environment, including Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)), Distributed Distributional Deep Deterministic Policy Gradient ([D4PG](https://arxiv.org/pdf/1804.08617.pdf)), Poximal Policy Optimization ([PPO](https://arxiv.org/pdf/1707.06347.pdf)) and Twin Delayed Deep Deterministic Policy Gradients ([TD3](https://arxiv.org/abs/1802.09477)). We will use DDPG for our base implementation, but *work in progress* versions of many of the remaining policies are also provided in this repo.  

This DDPG algorithm successfully trained in 295 episodes as determined by a running average of the scores of previous 100 episodes over 30.0.  

![alt text](https://github.com/cipher813/rl_continuous_control/blob/master/charts/201901030755_plotresults.png "Reacher20 Results with DDPG")

**DDPG**

DDPG was introduced by DeepMind in 2016 as an adaptation of Deep Q-Learning (DQN) to the continuous action domain.  The algorithm is described as an "actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces."  While DQN solves problems with high-dimensional observation (state) spaces, it can only handle discrete, low-dimensional action spaces.  

For the multi-agent implementation, we can [share experience amongst agents to accelerate learning](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html).  We do this by using the same memory ReplayBuffer for all agents.

The base model in this repo utilizes the DDPG algorithm.  

**D4PG**

D4PG was introduced by DeepMind as a conference paper for the [International Conference on Learning Representations (ICLR) 2018](https://iclr.cc/archive/www/doku.php%3Fid=iclr2018:main.html).  This modifies the DDPG algorithm with the use of a distributional version of the critic update, as well as use of N-step returns and prioritized experience replay.   

A work in progress version of this algorithm is also available in this repo.   

**PPO**

PPO was introduced by OpenAI in 2017 as an algorithm which alternates between "sampling data through interaction with the environment, and optimizing a 'surrogate' objective function using stochastic gradient ascent."

**TD3**

Published by McGill University and the University of Amsterdam for the [International Conference on Machine Learning (ICML) 2018](https://icml.cc/Conferences/2018), TD3 seeks to minimize the effects of overestimated value estimates from function approximation errors.  Building from DDPG, the algorithm takes the minimum value between a pair of critics to limit overestimation, and delays policy updates to reduce per-update error.  

<a name="hyperparameters"></a>
## Hyperparameters

Hyperparameters are found in the same file as the implementation in which it is deployed.  For [DDPG](https://github.com/cipher813/rl_continuous_control/blob/master/scripts/agents/DDPG.py), key hyperparameters include:

**Buffer Size.**  The ReplayBuffer memory size, as in the number of experiences that are remembered.   

**Batch Size.**  The size of each training batch sampled at a time.    

**Gamma.**  Discount factor for discounting past experiences (most recent experiences more highly rewarded as in less discount is applied).  

**Tau.**  For soft update of target parameters.  

**Learning Rate (Actor and Critic).**  Learning rates of actor and critic.  

**Weight Decay.**  L2 weight decay (not used with value of 0.)

<a name="network"></a>
# Neural Network Architecture

An actor-critic agent uses function approximation to learn both a policy &pi; (actor) and a value function V (critic which learns to evaluate V<sub>&pi;</sub> using TD estimate).  The algorithm runs as follows:
1. input state into actor and output the distribution over actions to take in that state &pi;(a|s;&theta;<sub>&pi;</sub>), returning experience (s, a, r, s').
2. train critic using TD estimate of r + &gamma;V(s'; &theta;<sub>v</sub>) to obtain state function of policy V(s;&theta;<sub>v</sub>).
3. calculate advantage function A(s,a) = r + &gamma;V(s';&theta;<sub>v</sub>) - V(s;&theta;<sub>v</sub>)
4. Train actor using calculated advantage as baseline.  

Where a is action, s is state, s' is next state, V is value, A is advantage, &pi; is policy, &theta; is neural network weights and &gamma; is discount variable.  

The DDPG algorithm utilizes a pair of neural networks, for each the actor and critic.  Common to both are a three layer neural network, receiving the state size of 33 variables corresponding to position, rotation, velocity and angular velocities of the arm as input.  The two layers are made up of 400 and 300 nodes.  Adam is used as the optimizer, and ReLU is used as the per-layer activations.  

The **actor network** uses a tanh output layer mapping to distribution over action size vector of 4 numbers, corresponding to torque applicable to two joints, where each number is between -1 and 1.  

The **critic network** is batch normalized and outputs a single value state policy function.

<a name="nextsteps"></a>
# Next Steps

Ideally I would like to make this script functional with several algorithms and environments.  While work in progress implementations of algorithms and basic functionality for other environments are included in this repo, they all need to be finished and tested.  Only DDPG for Reacher20 works moderately well at this point.  

Potential areas to explore in further work include:

**Algorithm implementations**

Complete extension of this script to cover similar, and potentially more stable and better-performing, algorithms, such as D4PG, PPO, A2C and TD3.  

**Environments**

Complete interchangeability of the script to run both Reacher1 and Reacher20, as well as environments from other platforms, such as OpenAI gym.  
