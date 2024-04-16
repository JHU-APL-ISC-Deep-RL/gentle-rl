import numpy as np
import torch
import gzip
import pickle
from box import Box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OffPolicyBuffer(object):

    def __init__(self, capacity, obs_dim, act_dim, store_costs=False):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # need for when episode is truncated
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32) if store_costs else None
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.mean_obs = np.zeros((1, obs_dim), dtype=np.float32)
        self.std_obs = np.ones((1, obs_dim), dtype=np.float32)
        self.raw_data = []
        self.episode_rewards = [0]
        self.episode_lengths = [0]
        self.episode_costs = [0]
        self.index, self.size, self.capacity, self.store_costs = 0, 0, capacity, store_costs

    def update(self, obs, action, reward, next_obs, terminated, info=None):
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminated[self.index] = terminated
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.episode_rewards[-1] += reward
        self.episode_lengths[-1] += 1
        if info is not None and 'cost' in info:
            self.episode_costs[-1] += info['cost']
            if self.store_costs:
                self.costs[self.index] = info['cost']

    def trajectory_update(self, trajectory_buffer, next_obs):
        """  Update buffer using a full trajectory (needed for Conservative Safety Critic)  """
        # todo: may need to update for 1-d obs, action space (don't want vstack there)
        steps = trajectory_buffer.shape[0]
        self.size = min(self.size + steps, self.capacity)
        if self.index + steps <= self.capacity:
            self.obs[self.index: self.index + steps] = np.vstack(trajectory_buffer[:, 0])
            self.next_obs[self.index: self.index + steps - 1] = np.vstack(trajectory_buffer[1:, 0])
            self.next_obs[self.index + steps - 1] = next_obs
            self.actions[self.index: self.index + steps] = np.vstack(trajectory_buffer[:, 1])
            self.rewards[self.index: self.index + steps] = trajectory_buffer[:, 2]
            self.terminated[self.index: self.index + steps] = trajectory_buffer[:, 6]
            if self.store_costs:
                self.costs[self.index: self.index + steps] = trajectory_buffer[:, 7]
            self.index += steps
        else:  # wrap
            part_1 = self.capacity - self.index
            part_2 = self.index + steps - self.capacity
            self.obs[self.index:] = np.vstack(trajectory_buffer[:part_1, 0])
            self.obs[:part_2] = np.vstack(trajectory_buffer[part_1:, 0])
            self.next_obs[self.index:] = np.vstack(trajectory_buffer[1:part_1+1, 0])
            self.next_obs[:part_2 - 1] = np.vstack(trajectory_buffer[part_1+1:, 0])
            self.next_obs[part_2 - 1] = next_obs
            self.actions[self.index:] = np.vstack(trajectory_buffer[:part_1, 1])
            self.actions[:part_2] = np.vstack(trajectory_buffer[part_1:, 1])
            self.rewards[self.index:] = trajectory_buffer[:part_1, 2]
            self.rewards[:part_2] = trajectory_buffer[part_1:, 2]
            self.terminated[self.index:] = trajectory_buffer[:part_1, 6]
            self.terminated[:part_2] = trajectory_buffer[part_1:, 6]
            if self.store_costs:
                self.costs[self.index:] = trajectory_buffer[:part_1, 7]
                self.costs[:part_2] = trajectory_buffer[part_1:, 7]
            self.index = part_2

    def sample(self, batch_size, recent=None):
        if recent is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            if self.index >= recent:
                ind = np.arange(self.index-recent, self.index)
            else:
                ind = np.arange(self.index)
                if self.size == self.capacity:  # buffer has wrapped
                    ind = np.concatenate((ind, np.arange(self.capacity - recent + self.index, self.capacity)))
            indices = np.random.choice(ind, size=batch_size)
        sample = {'obs': self.obs[indices],
                  'next_obs': self.next_obs[indices],
                  'actions': self.actions[indices],
                  'rewards': self.rewards[indices],
                  'terminated': self.terminated[indices]}
        if self.store_costs:
            sample.update({'costs': self.costs[indices]})
        return Box({k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in sample.items()})

    def reset_episode(self):
        self.episode_rewards += [0]
        self.episode_lengths += [0]
        self.episode_costs += [0]

    def reset_logging(self):
        self.episode_rewards = [self.episode_rewards[-1]]
        self.episode_lengths = [self.episode_lengths[-1]]
        self.episode_costs = [self.episode_costs[-1]]

    def compress(self):
        """  Compress buffers for storage (bigger effect from file compression) """
        state = {}
        for key, value in self.__dict__.items():
            if key in ['obs', 'actions', 'rewards', 'next_obs', 'costs', 'terminated']:
                state[key] = value[:self.size]
            else:
                state[key] = value
        return state

    def save(self, filename):
        """  Save in compressed form"""
        state = self.compress()  # trim excess indices
        # save state:
        with gzip.open(filename, 'wb') as fp:
            pickle.dump(state, fp)

    def load(self, filename):
        """  Load stored buffer  """
        # load state:
        with gzip.open(filename, 'rb') as fp:
            state = pickle.load(fp)
        # distribute values:
        self.size = state['size']
        if state['size'] > self.capacity:
            self.expand(state['size'])
        for key, value in state.items():
            if key in ['obs', 'actions', 'rewards', 'next_obs', 'terminated']:
                self.__dict__[key][:state['size']] = value
            elif key == 'costs':
                if self.store_costs:
                    self.__dict__['costs'][:state['size']] = state['costs']
            else:
                self.__dict__[key] = value

    def expand(self, capacity):
        """  Expand capacity to load a data set that is larger than current capacity  """
        obs_dim = self.obs.shape[1]
        act_dim = self.actions.shape[1]
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32) if self.store_costs else None
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity

    def store_raw(self, buffer, bootstrap):
        self.raw_data.append([buffer, bootstrap])


class RecentCostOffPolicyBuffer(OffPolicyBuffer):
    def __init__(self, capacity, obs_dim, act_dim, recent=10000):
        super().__init__(capacity, obs_dim, act_dim, True)
        self.recent = recent
        self.recent_costs = []

    def update(self, obs, action, reward, next_obs, terminated, info=None):
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminated[self.index] = terminated
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.episode_rewards[-1] += reward
        self.episode_lengths[-1] += 1
        if info is not None and 'cost' in info:
            self.episode_costs[-1] += info['cost']
            if self.store_costs:
                self.costs[self.index] = info['cost']
                self.recent_costs.append(info['cost'])
                if len(self.recent_costs) > self.recent:
                    self.recent_costs.pop(0)


class RewardOffPolicyBuffer(OffPolicyBuffer):
    def __init__(self, capacity, obs_dim, act_dim, store_costs=False):
        super().__init__(capacity, obs_dim, act_dim, store_costs)
        self.ep_rewards = np.zeros(capacity, dtype=np.float32)
        self.ep_start = 0

    def sample(self, batch_size, recent=None):
        if self.size < self.capacity:
            indices = np.random.randint(0, self.ep_start, size=batch_size)
        else:
            if self.ep_start <= self.index:
                ind = np.concatenate((np.arange(self.ep_start), np.arange(self.index, self.capacity)))
            else:  # wrapped
                ind = np.arange(self.index, self.ep_start)
            indices = np.random.choice(ind, size=batch_size, replace=False)
        sample = {'obs': self.obs[indices],
                  'next_obs': self.next_obs[indices],
                  'actions': self.actions[indices],
                  'rewards': self.rewards[indices],
                  'terminated': self.terminated[indices],
                  'ep_rewards': self.ep_rewards[indices]}
        if self.store_costs:
            sample.update({'costs': self.costs[indices]})
        return Box({k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in sample.items()})

    def reset_episode(self):
        if self.index >= self.ep_start:
            self.ep_rewards[self.ep_start: self.index] = self.episode_rewards[-1]
        else:  # buffer has wrapped
            self.ep_rewards[:self.index] = self.episode_rewards[-1]
            self.ep_rewards[self.ep_start:] = self.episode_rewards[-1]
        self.ep_start = self.index
        super().reset_episode()


class OnPolicyBuffer(object):
    """  On-policy experience buffer.  Not all fields are used for all types of learning.  """
    def __init__(self):
        self.observations = np.array([])
        self.bootstrap_obs = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])
        self.values = np.array([])
        self.bootstraps = np.array([])
        self.dones = np.array([])
        self.q_values = np.array([])
        self.advantages = np.array([])
        self.policies = np.array([])
        self.log_probs = np.array([])
        self.utilities = np.array([])
        self.costs = np.array([])
        self.c_q_values = np.array([])
        self.c_utilities = np.array([])
        self.cost_values = np.array([])
        self.cost_advantages = np.array([])
        self.steps = 0
        self.trajectories = 0
        self.episode_rewards = [0]
        self.episode_lengths = [0]
        self.episode_costs = [0]

    def update(self, trajectory_buffer, q_values=None, advantages=None, utilities=None,
               c_q_values=None, c_utilities=None):
        """  Add a new experience to the buffer  """
        self.steps += trajectory_buffer.shape[0]
        if len(self.observations) == 0:
            self.observations = np.vstack(trajectory_buffer[:, 0])
            self.actions = np.vstack(trajectory_buffer[:, 1])
            self.policies = np.vstack(trajectory_buffer[:, 3])
        else:
            self.observations = np.concatenate((self.observations, np.vstack(trajectory_buffer[:, 0])))
            self.actions = np.concatenate((self.actions, np.vstack(trajectory_buffer[:, 1])))
            self.policies = np.concatenate((self.policies, np.vstack(trajectory_buffer[:, 3])))
        self.rewards = np.concatenate((self.rewards, trajectory_buffer[:, 2]))
        self.log_probs = np.concatenate((self.log_probs, trajectory_buffer[:, 4]))
        self.values = np.concatenate((self.values, trajectory_buffer[:, 5]))
        self.dones = np.concatenate((self.dones, trajectory_buffer[:, 6]))
        if trajectory_buffer.shape[1] > 7:
            self.costs = np.concatenate((self.costs, trajectory_buffer[:, 7]))
        self.trajectories = np.sum(self.dones > 0)  # gets both terminated, truncated
        if q_values is not None:
            self.q_values = np.concatenate((self.q_values, q_values))  # only needed in training
        if advantages is not None:  # computation of advantage may not be in trajectory runner
            self.advantages = np.concatenate((self.advantages, advantages))
        if utilities is not None:  # don't need this for standard learning
            self.utilities = np.concatenate((self.utilities, utilities))
        if c_q_values is not None:
            self.c_q_values = np.concatenate((self.c_q_values, c_q_values))  # only needed in constrained training
        if c_utilities is not None:  # don't need this for standard learning
            self.c_utilities = np.concatenate((self.c_utilities, c_utilities))

    def log_bootstrap_obs(self, obs):
        if len(self.bootstrap_obs) == 0:
            self.bootstrap_obs = obs[np.newaxis, ...]
        else:
            self.bootstrap_obs = np.concatenate((self.bootstrap_obs, obs[np.newaxis, ...]))

    def reset_episode(self):
        self.episode_rewards += [0]
        self.episode_lengths += [0]
        self.episode_costs += [0]

    def reset_logging(self):
        self.episode_rewards = [self.episode_rewards[-1]]
        self.episode_lengths = [self.episode_lengths[-1]]
        self.episode_costs = [self.episode_costs[-1]]

    def update_episode(self, reward, cost=None):
        self.episode_rewards[-1] += reward
        if cost is not None:
            self.episode_costs[-1] += cost
        self.episode_lengths[-1] += 1