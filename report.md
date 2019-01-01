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

In this project, we explored a variety of policies to solve this continuous state space environment, including Deep Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971), Distributed Distributional Deep Deterministic Policy Gradient [D4PG](https://arxiv.org/pdf/1804.08617.pdf), Poximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf) and Twin Delayed Deep Deterministic Policy Gradients [TD3](https://arxiv.org/abs/1802.09477). We will use DDPG for our base implementation, but work in progress of the remaining policies are also available in this repo.  

**DDPG**

For the multi-agent implementation, we can [share experience amongst agents to accelerate learning](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html).  We do this by using the same memory ReplayBuffer for all agents.

**D4PG**


**PPO**


**TD3**

<a name="hyperparameters"></a>
## Hyperparameters

Hyperparameters used in this implementation include:
[xxx]

<a name="network"></a>
# Neural Network Architecture

The underlying Actor Critic Network used in the implementation is a [x] layer network with the input and output layers mapping to the state and action sizes of 33 and 4 (per [Environment](#environment)), respectively.  There are [xxx].  

<a name="nextsteps"></a>
# Next Steps

Potential areas to explore in further work include:

**Algorithm implementations**

[xxx]


**Environments**

[xxx]
