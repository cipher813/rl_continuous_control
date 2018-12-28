"""
Code inspired by D4PG implmentation at
https://github.com/kelvin84hk/DRLND_P2_Continuous_Control/
"""
import copy
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 3e-3        # learning rate of the critic
WEIGHT_DECAY = 0.#0.01     # L2 weight decay
VMAX = 5
VMIN = 0
ATOMS = 51
DELTA_Z = (VMAX - VMIN) / (ATOMS - 1)
# LEARN_EVERY = 20
# NUM_LEARN = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class D4PG:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed=7):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.learn_every = num_agents # UPDATE_EVERY
        self.num_learn = num_agents//2 if num_agents > 1 else 1 # N_step
        self.seed = random.seed(random_seed)
        self.GAMMA = GAMMA

        self.rewards_queue = deque(maxlen=self.num_learn)
        self.states_queue = deque(maxlen=self.num_learn)

        self.timestep = 0
        self.train_start = 2000

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, random_seed).to(device) #max_action,
        self.actor_target = Actor(state_size, action_size, random_seed).to(device) #max_action,
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size, action_size, random_seed, atoms=ATOMS, vmin=VMIN, vmax=VMAX).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, atoms=ATOMS, vmin=VMIN, vmax=VMAX).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = Memory(BUFFER_SIZE)
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # if self.num_agents>1:
        #     for i in range(self.num_agents):
        #         self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
        # else:
        #     self.memory.add(state, action, reward, next_state, done)

        self.states_queue.appendleft([state,action])
        for i in range(self.num_agents):
            self.rewards_queue.appendleft(reward[i]*GAMMA**self.num_learn)
        for i in range(len(self.rewards_queue)):
            self.rewards_queue[i] = self.rewards_queue[i]/GAMMA

        if len(self.rewards_queue)>=self.num_learn:
            temps = self.states_queue.pop()
            state = torch.tensor(temps[0]).float().to(device)
            next_state = torch.tensor(next_state).float().to(device)
            action = torch.tensor(temps[1]).float().unsqueeze(0).to(device)

            self.critic.eval()
            with torch.no_grad():
                Q_expected = self.critic(state, action)
            self.critic.train()
            self.actor_target.eval()
            with torch.no_grad():
                action_next = self.actor_target(next_state)
            self.actor_target.train()
            self.critic_target.eval()
            with torch.no_grad():
                Q_target_next = self.critic_target(next_state, action_next)
                Q_target_next = F.softmax(Q_target_next, dim=1)
            self.critic_target.train()
            sum_reward = torch.tensor(sum(self.rewards_queue)).float().unsqueeze(0).to(device)
            done_temp = torch.tensor(done).float().to(device)
            Q_target_next = self.distr_projection(Q_target_next, sum_reward, done_temp, GAMMA**self.num_learn)
            Q_target_next = -F.log_softmax(Q_expected, dim=1)*Q_target_next
            error = Q_target_next.sum(dim=1).mean().cpu().data

            state = state.cpu().data.numpy()
            next_state = next_state.cpu().data.numpy()
            action=action.squeeze(0).cpu().data.numpy()
            self.memory.add(error, (state, action, sum(self.rewards_queue), next_state, done))
            self.rewards_queue.pop()
            if done:
                self.states_queue.clear()
                self.rewards_queue.clear()

        self.timestep = (self.timestep + 1)% self.num_learn
        if timestep==0:
            if self.memory.tree.n_entries > self.train_start:
                batch_not_ok = True
                while batch_not_ok:
                    mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
                    mini_batch = np.array(mini_batch).transpose()
                    if mini_batch.shape==(5,BATCH_SIZE):
                        batch_not_ok = False
                    else:
                        print(mini_batch.shape)
                try:
                    states = np.vstack([m for m in mini_batch[0] if m is not None])
                except:
                    print("States not same dim")
                    pass
                try:
                    actions = np.vstack([m for m in mini_batch[1] if m is not None])
                except:
                    print("Actions not same dim")
                    pass
                try:
                    rewards = np.vstack([m for m in mini_batch[2] if m is not None])
                except:
                    print("Rewards not same dim")
                    pass
                try:
                    next_states = np.vstack([m for m in mini_batch[3] if m is not None])
                except:
                    print("Next States not same dim")
                    pass
                try:
                    dones = np.vstack([m for m in mini_batch[4] if m is not None])
                except:
                    print("Dones not same dim")
                    pass
                dones = dones.astype(int)

                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).float().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                dones = torch.from_numpy(dones).float().to(device)
                experiences = (states, actions, rewards, next_states, dones)
                self.learn(experiences, GAMMA, idxs)

        # Save experience / reward
        # if self.num_agents>1:
        #     for i in range(self.num_agents):
        #         self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
        # else:
        #     self.memory.add(state, action, reward, next_state, done)

        # learn 10 (NUM_LEARN) experiences every 20 (LEARN_EVERY) timesteps
        # if timestep%self.learn_every==0:
        #     if len(self.memory)>BATCH_SIZE:
        #         for i in range(self.num_learn):
        #             experiences = self.memory.sample()
        #             self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True): # act
        """Returns actions for given state as per current policy."""
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # return self.actor(state).cpu().data.numpy().flatten()
        state = torch.tensor(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.squeeze(np.clip(action,-1.0,1.0))

        # state = torch.from_numpy(state).float().to(device)
        # self.actor.eval()
        # with torch.no_grad():
        #     action = self.actor(state).cpu().data.numpy()
        # self.actor.train()
        # if add_noise:
        #     if self.num_agents>1:
        #         for i in range(self.num_agents):
        #             action[i] += self.noise.sample()
        #     else:
        #         action += self.noise.sample()
        # return np.clip(action, -1, 1)

    def distr_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma):
        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, ATOMS), dtype=np.float32)
        dones_mask = np.squeeze(dones_mask)
        rewards = rewards.reshape(-1)

        for atom in range(ATOMS):
            tz_j = np.minimum(VMAX, np.maximum(VMIN, rewards + (VMIN + atom * DELTA_Z)*gamma))
            b_j = (tz_j - VMIN) / DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l

            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l

            proj_distr[ne_mask,l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask]=0.0
            tz_j = np.minimum(VMAX, np.maximum(VMIN, rewards[dones_mask]))
            b_j = (tz_j - VMIN) / DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            if dones_mask.shape==():
                if dones_mask:
                    proj_distr[0,l]=1.0
                else:
                    ne_mask = u != l
                    proj_distr[0,l]=(u-b_j)[ne_mask]
                    proj_distr[0,u]=(b_j-l)[ne_mask]
            else:
                eq_dones = dones_mask.copy()

                eq_dones[dones_mask] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l[eq_mask]]=1.0
                ne_mask = u != l
                ne_dones = dones_mask.copy()
                ne_dones[dones_mask] = ne_mask
                if ne_dones.any():
                    proj_distr[ne_dones, l[ne_mask]] = (u-b_j)[ne_mask]
                    proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr).to(device)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, idxs=None): # train
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_expected = self.critic(states, actions) # current_Q
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next) # target_Q
        # # Compute Q targets for current states (y_i)
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # target_Q
        # Compute critic loss
        Q_targets_next = F.softmax(Q_targets_next,dim=1)
        Q_targets_next = self.distr_projection(Q_targets_next, rewards, dones, gamma*self.num_learn)
        Q_targets_next = -F.log_softmax(Q_expected, dim=1)*Q_targets_next
        critic_loss = Q_targets_next.sum(dim=1).mean()

        self.critic.eval()
        with torch.no_grad():
            errors = Q_targets_next.sum(dim=1).cpu().data.numpy()
        self.critic.train()

        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(),1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)

        crt_distr_v = self.critic(states, actions_pred)
        actor_loss = -self.critic.distr_to_q(crt_distr_v)
        actor_loss = actor_loss.mean()

        # actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        # nn.utils.clip_grad_norm_(self.actor.parameters(),1)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # def save(self, filename, directory):
    #     torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    #     torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    #
    #
    # def load(self, filename, directory):
    #     self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    #     self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 1e-3

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1)//2
        self.tree[parent]+= change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write +=1
        if self.write>= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0,s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#
#     def __init__(self, action_size, buffer_size, batch_size, seed):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)
#
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.memory, k=self.batch_size)
#
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
#
#         return (states, actions, rewards, next_states, dones)
#
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300): #max_action,
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)

        # self.bn1 = nn.BatchNorm1d(fc1_units) # removing for gym

        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # self.bn = nn.BatchNorm1d(action_size)
        self.tanh=nn.Tanh()

        # self.max_action = max_action
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # x = F.relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        # x = self.bn(x)
        return self.tanh(x)

        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1=400, fc2=300, fc3=300, fc4=300, atoms=51, vmin=-1, vmax=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)

        # self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc2 = nn.Linear(fc1+action_size, fc2)
        self.fc3 = nn.Linear(fc2, atoms)
        delta = (vmax - vmin) / (atoms - 1)
        self.register_buffer("supports", torch.arange(vmin, vmax+delta, delta))
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fc1(state))

        # xs = F.relu(self.bn1(self.fc1(state)))
        x = torch.cat((xs, action),dim=1)

        x = F.leaky_relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        return self.fc3(x)

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1)* self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
