# Reinforcement Learning Continuous Control

## Repo Table of Contents

[Project Overview](#overview)

[Environment Setup](#setup)

[Model](#model)

[Resources](#resources)

[Report](https://github.com/cipher813/rl_continuous_control/blob/master/report.md#report)

- [RL Environment](https://github.com/cipher813/rl_continuous_control/blob/master/report.md#environment)

- [Algorithm](https://github.com/cipher813/rl_continuous_control/blob/master/report.md#algorithm)

- [Hyperparameters](https://github.com/cipher813/rl_continuous_control/blob/master/report.md#hyperparameters)

- [Network Architecture](https://github.com/cipher813/rl_continuous_control/blob/master/report.md#network)

- [Next Steps](https://github.com/cipher813/rl_continuous_control/blob/master/report.md#nextsteps)

<a name="overview"></a>
## Project Overview

For [Project 2](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with teaching an agent to maintain a target position using the "Reacher" environment configured by Udacity on [Unity's ML-Agents platform](https://github.com/Unity-Technologies/ml-agents).  

For further information on the environment, see the accompanying project [Report](https://github.com/cipher813/rl_continuous_control/blob/master/report.md) or Udacity's [project github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).  

In this project, we explored a variety of policies to solve this continuous state space environment, including Deep Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971), Distributed Distributional Deep Deterministic Policy Gradient [D4PG](https://arxiv.org/pdf/1804.08617.pdf), Poximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf) and Twin Delayed Deep Deterministic Policy Gradients [TD3](https://arxiv.org/abs/1802.09477). We will use DDPG for our base implementation, but work in progress of the remaining policies are also available in this repo.   

The algorithms are further explained in the accompanying [Report](https://github.com/cipher813/rl_continuous_control/blob/master/report.md).

<a name="setup"></a>
## Environment Setup

To set up the python (conda) environment, in the root directory, type:

`conda env update --file=environment_drlnd.yml`

This requires installation of [OpenAI Gym](https://github.com/openai/gym) and Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents).   

In the root directory, run `python setup.py` to set up directories and download specified environments.  When running this file, make sure you have the full path to your root repo folder readily available (and end the input with a "/").

If you need to further review and access environment implementation, visit the project repo [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

<a name="model"></a>
## The Model

The key files in this repo include:

### Scripts

[main.py](https://github.com/cipher813/rl_continuous_control/tree/master/scripts)
Execute this script to train in the environment(s) and agent(s) specified on this script in the environment and agent dictionaries, respectively.

[util.py](https://github.com/cipher813/rl_continuous_control/tree/master/scripts)
Contains functions to train in Unity and OpenAI environments, and to chart results.

[agents](https://github.com/cipher813/rl_continuous_control/tree/master/scripts) folder
Contains agent classes as specified policies.  See [report.md](https://github.com/cipher813/rl_continuous_control/blob/master/report.md) for additional details on agent implementations.

To train the agent, first open main.py in your favorite text editor (ie `nano main.py` or `vi main.py`).  Make sure the path to the root repo folder is correct and that the proper environments and agents (policies) are selected.  Then, in the command line run:

`source activate drlnd` # to activate python (conda) environment
`python main.py` # to train the environment and agent (policy)

### Notebooks

[rl2_results.ipynb](https://github.com/cipher813/rl_continuous_control/tree/master/notebooks)

Charts the results from specified results dictionary pickle file.  

### Results

Contains the "checkpoint" [model weights](https://github.com/cipher813/rl_continuous_control/tree/master/results) of each implementation.  

<a name="resources"></a>
## Resources

The algorithms used in this project were inspired by a variety of sources and
authors, including implementations from the following github handles:

[Udacity](https://github.com/udacity/deep-reinforcement-learning)

[Kinwo](https://github.com/kinwo/deeprl-continuous-control)

[partha746](https://github.com/partha746/DRLND_P2_Reacher_EnV)

[kelvin84hk](https://github.com/kelvin84hk/DRLND_P2_Continuous_Control/)

[sperazza](https://github.com/sperazza/MultiAgentDeepRL/)
