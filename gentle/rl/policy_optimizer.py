import sys
import os
import json
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from box import Box
from shutil import rmtree
from mpi4py import MPI
from copy import deepcopy
from gentle.common.loggers import LoggerMPI
from gentle.common.buffers import OnPolicyBuffer
from gentle.common.utils import get_env_object, get_network_object, get_sampler
from gentle.common.samplers import compute_entropy
from gentle.common.mpi_data_utils import mpi_sum, average_grads, sync_weights, mpi_statistics_scalar, \
    mpi_avg, collect_dict_of_lists, mpi_gather_objects, print_now, print_zero


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class PolicyOptimizer(object):

    def __init__(self, config):
        """
        Policy Optimization Agent that either uses policy gradient or surrogate objective in conjunction
        with configurable variance reduction measures.
        """
        self.id = MPI.COMM_WORLD.Get_rank()
        self.config = config
        self.process_config()
        self.mode = ''
        self.env = None
        self.obs = None
        self.pi_network = None
        self.v_network = None
        self.pi_optimizer = None
        self.v_optimizer = None
        self.sampler = None
        self.buffer = None
        self.logger = None
        self.epsilon = 1.e-6
        num_workers = MPI.COMM_WORLD.Get_size()
        torch.manual_seed((self.id + 1 + self.config.seed * num_workers) * 2000)
        np.random.seed((self.id + 1 + self.config.seed * num_workers) * 5000)

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate  """
        # General training configuration:
        self.config.setdefault('seed', 0)                     # Random seed parameter
        self.config.setdefault('training_frames', int(5e7))   # Number of frames to use for training run
        self.config.setdefault('max_ep_length', -1)           # Maximum episode length (< 0 means no max)
        self.config.setdefault('batch_size', 30000)           # Average number of experiences to base an update on
        self.config.setdefault('pi_lr', .0003)                # Policy optimizer learning rate
        self.config.setdefault('v_lr', .001)                  # Value optimizer learning rate
        self.config.setdefault('train_v_iter', 80)            # Value updates per epoch
        self.config.setdefault('train_pi_iter', 80)           # Policy updates per epoch
        self.config.setdefault('gamma', 0.99)                 # Discount factor gamma
        self.config.setdefault('lam', 0.95)                   # GAE factor lambda
        self.config.setdefault('clip', 0.2)                   # Clip factor for policy (< 0 means none)
        self.config.setdefault('v_clip', -1)                  # Clip factor for value function (< 0 means none)
        self.config.setdefault('max_kl', -1)                  # KL criteria for early stopping (< 0 means ignore)
        # TRPO-specific configuration:
        self.config.setdefault('trpo', False)
        if self.config.trpo:
            self.config.train_pi_iter = 1  # Policy updates per epoch
            self.config.clip = -1  # Clip factor for policy (< 0 means none)
            self.config.reward_to_go = True  # Whether to use reward-to-go
            self.config.gae = True  # Whether to use generalized advantage estimation
            self.config.setdefault('surrogate', True)  # Whether to use surrogate objective
            self.config.setdefault('cg_iter', 10)  # Number of iterations in conjugate gradient method
            self.config.setdefault('cg_delta', 0)  # Early stopping in conjugate gradient solver
            self.config.setdefault('damping_coeff', 0.1)  # Improves numerical stability of hessian vector product
            self.config.setdefault('backtrack_iter', 10)  # Maximum number of backtracks allowed per line search
            self.config.setdefault('backtrack_coeff', 0.8)  # How far back to step during backtracking line search
            self.config.backtrack_ratios = self.config.backtrack_coeff ** np.arange(self.config.backtrack_iter)
        # Variance reduction measures:
        self.config.setdefault('reward_to_go', True)          # Whether to use reward-to-go
        self.config.setdefault('gae', True)                   # Whether to use generalized advantage estimation
        self.config.setdefault('surrogate', False)            # Whether to use surrogate objective
        if self.config.gae or self.config.surrogate:
            assert self.config.reward_to_go, "Given rest of configuration, must use reward-to-go."
            assert 'v_network' in self.config, "Given rest of configuration, must include value network."
        self.config.setdefault('full_kl', True)               # Whether to use full KL estimation or approximation
        self.config.setdefault('bound_corr', False)           # Whether to use boundary correction in policy logprob
        self.config.pi_network.bound_corr = self.config.bound_corr
        # Testing configuration:
        self.config.setdefault('test_iteration', -1)          # Test latest network
        self.config.setdefault('test_episodes', 1000)
        self.config.setdefault('test_random_base', 100000)
        self.config.test_random_base = self.config.test_random_base*(self.config.seed + 1)
        self.config.setdefault('render', False)
        # Logging and storage configurations:
        self.config.setdefault('log_info', True)
        self.config.setdefault('checkpoint_every', int(1e7))
        self.config.setdefault('evaluation_every', -1)        # By default, don't run evaluations
        self.config.setdefault('evaluation_episodes', int(self.config.batch_size / self.config.max_ep_length))
        self.config.setdefault('use_prior_nets', False)       # Whether to pick up where previous training left off
        self.config.setdefault('model_folder', '../../output/rl_training')
        self.config.setdefault('log_folder', '../../logs/rl_training')
        self.config.model_folder = os.path.join(os.getcwd(), self.config.model_folder)
        self.config.log_folder = os.path.join(os.getcwd(), self.config.log_folder)
        self.config.model_folder = self.config.model_folder + '_' + str(self.config.seed)
        self.config.log_folder = self.config.log_folder + '_' + str(self.config.seed)
        if sys.platform[:3] == 'win':
            self.config.model_folder = self.config.model_folder.replace('/', '\\')
            self.config.log_folder = self.config.log_folder.replace('/', '\\')
        if self.id == 0:
            if not self.config.use_prior_nets:  # start a fresh training run
                if os.path.isdir(self.config.log_folder):
                    rmtree(self.config.log_folder, ignore_errors=True)
                if os.path.isdir(self.config.model_folder):
                    rmtree(self.config.model_folder, ignore_errors=True)
            Path(self.config.model_folder).mkdir(parents=True, exist_ok=True)
            Path(self.config.log_folder).mkdir(parents=True, exist_ok=True)

    def train(self):
        """  Train neural network  """
        # Initialize relevant objects:
        self.mode = 'train'
        self.initialize_env()
        last_checkpoint = self.initialize_networks()
        self.sampler = get_sampler(self.config.pi_network)
        self.initialize_optimizers()
        self.initialize_logging()
        # Run training:
        total_steps = max(last_checkpoint * self.config.checkpoint_every, 0)
        last_evaluation = total_steps // self.config.evaluation_every if total_steps > 0 else -1
        while total_steps < self.config.training_frames:
            # Collect data:
            self.pi_network.train()
            if self.v_network is not None:
                self.v_network.train()
            self.buffer = OnPolicyBuffer()  # reset experience buffer
            steps_current, mean_steps, step_tracker, all_episode_summaries = 0, 0, [0], {}
            while steps_current < self.config.batch_size - mean_steps / 2:
                episode_summary = self.run_trajectory()
                steps_current = mpi_sum(MPI.COMM_WORLD, self.buffer.steps)
                step_tracker.append(steps_current - sum(step_tracker))
                mean_steps = sum(step_tracker[1:])/(len(step_tracker[1:]))
                if steps_current == 0:  # first iteration
                    all_episode_summaries = {k: [v] for k, v in episode_summary.items()}
                else:
                    all_episode_summaries = self.concatenate_dict_of_lists(all_episode_summaries, episode_summary)
            previous_steps = total_steps
            total_steps += steps_current
            # Update network(s), as prescribed by config:
            losses = self.update_network()
            # Update logging, save the model:
            local_steps = self.buffer.steps
            if total_steps // self.config.evaluation_every > last_evaluation:
                evaluation = self.run_evaluation()
                last_evaluation += 1
            else:
                evaluation = None
            self.update_logging(all_episode_summaries, losses, evaluation, local_steps, previous_steps)
            if self.id == 0:
                last_checkpoint = self.save_networks(total_steps, last_checkpoint)

    def initialize_env(self):
        """  Initialize environment object.  """
        self.env = get_env_object(self.config)
        if self.mode == 'test':
            self.env.seed(self.id*self.config.test_episodes + self.config.test_random_base)
        self.obs, _ = self.reset_env()

    def initialize_networks(self):
        """  Initialize network objects  """
        last_checkpoint = -1
        self.pi_network = get_network_object(self.config.pi_network)
        if 'v_network' in self.config:
            self.v_network = get_network_object(self.config.v_network)
        if self.config.use_prior_nets:
            if self.mode == 'test' and self.config.test_iteration is not None:
                if self.config.test_iteration < 0:
                    self.pi_network.restore(os.path.join(self.config.model_folder, 'model-latest.pt'))
                    if self.v_network is not None:
                        self.v_network.restore(os.path.join(self.config.model_folder, 'value-latest.pt'))
                else:
                    self.pi_network.restore(os.path.join(self.config.model_folder,
                                                         'model-' + str(self.config.test_iteration) + '.pt'))
                    if self.v_network is not None:
                        self.v_network.restore(os.path.join(self.config.model_folder,
                                                            'value-' + str(self.config.test_iteration) + '.pt'))
            else:
                raise NotImplementedError('Currently, use of prior nets only supported in test mode.')
        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
        if self.v_network is not None:
            sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
        return last_checkpoint

    def initialize_optimizers(self):
        """  Initializes Adam optimizer for training network.  Only one worker actually updates parameters.  """
        self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.pi_lr)
        if self.v_network is not None:
            self.v_optimizer = torch.optim.Adam(params=self.v_network.parameters(), lr=self.config.v_lr)

    def initialize_logging(self):
        """  Initialize logger and store config (only on one process)  """
        with open(os.path.join(self.config.model_folder, 'config.pkl'), 'wb') as config_file:
            pickle.dump(self.config, config_file)  # store configuration
        self.logger = LoggerMPI(self.config.log_folder)
        self.logger.log_graph(self.obs, self.pi_network)

    def run_trajectory(self, random_seed=None):
        """  Run trajectories based on current network(s)  """
        episode_buffer, episode_info, entropies, num_frames = np.array([]).reshape(0, 7), {}, [], 0
        while True:
            policy, value = self.forward_pass()
            action, log_prob = self.sampler.get_action_and_log_prob(policy)
            trunc = num_frames == self.config.max_ep_length - 1
            output_obs, reward, terminated, truncated, info = self.step_env(action, trunc)
            if self.config.log_info:
                self.concatenate_dict_of_lists(episode_info, info)
            entropies += [compute_entropy(policy)]
            if self.config.render and self.mode == 'test':
                self.env.render()
            num_frames += 1
            episode_buffer = self.update_episode_buffer(episode_buffer, action, reward, policy,
                                                        log_prob, value, terminated + 2 * truncated)
            self.obs = output_obs
            if terminated or truncated:
                self.buffer.log_bootstrap_obs(self.obs)
                if random_seed is not None:
                    self.env.seed(random_seed)  # for testing
                self.obs, _ = self.reset_env()
                self.process_episode(episode_buffer, episode_info)
                break
        episode_summary = {'episode_reward': np.sum(episode_buffer[:, 2]), 'episode_length': num_frames,
                           'episode_mean_value': np.mean(episode_buffer[:, 5]) if 'v_network' in self.config else None,
                           'episode_entropy': np.mean(entropies), **{k: sum(v) for k, v in episode_info.items()}}
        return episode_summary

    def forward_pass(self):
        """  Runs forward pass of network(s).  For continuous action spaces, policy will be a tuple of mean, std. """
        if self.v_network is None:
            if 'pv_network' in self.config and self.config.pv_network:
                policy, value = self.pi_network.forward_with_processing(self.obs)
                value = value.detach().numpy()[0, 0]
            else:
                policy = self.pi_network.forward_with_processing(self.obs)
                value = None
        else:
            policy = self.pi_network.forward_with_processing(self.obs)
            value = self.v_network.forward_with_processing(self.obs)
            value = value.detach().numpy()[0]
        return policy, value

    def update_episode_buffer(self, episode_buffer, action, reward, policy, log_prob, value, done):
        """  Updates episode buffer for current step  """
        if self.pi_network.config.discrete:
            policy_to_store = np.squeeze(policy.detach().numpy())
        else:
            policy_to_store = np.concatenate((policy[0].detach().numpy(), policy[1].detach().numpy()))
        experience = np.reshape(np.array([self.obs, action, reward, policy_to_store, log_prob, value, done],
                                         dtype=object), (1, 7))
        return np.concatenate((episode_buffer, experience))

    def process_episode(self, episode_buffer, episode_info):
        """  Processes a completed episode, storing required data in buffer  """
        if self.mode == 'train':
            q_values = self.compute_target_values(episode_buffer[:, 2])
            self.buffer.update(episode_buffer, q_values)

    def compute_target_values(self, rewards):
        """  Computes value function targets (without bootstrapping)  """
        trajectory_length = rewards.shape[0]
        if not self.config.reward_to_go:  # Return full-episode discounted return at each time step
            indices = np.arange(trajectory_length)
            discounts = np.power(self.config.gamma, indices)
            discounted_rewards = rewards * discounts
            discounted_episode_reward = np.sum(discounted_rewards)
            target_values = np.ones(rewards.shape) * discounted_episode_reward
        else:
            target_values = np.zeros((trajectory_length,))
            for start in range(trajectory_length):
                indices = np.arange(start, trajectory_length)
                discounts = np.power(self.config.gamma, indices - start)
                discounted_future_rewards = rewards[start:] * discounts
                target_values[start] = np.sum(discounted_future_rewards)
        return target_values

    def update_network(self):
        """  Updates the networks based on processing from all workers  """
        self.pi_network.eval()
        if self.v_network is not None:
            self.v_network.eval()
        # Update value network:
        observations = torch.from_numpy(np.vstack(self.buffer.observations)).float()
        values = torch.from_numpy(self.buffer.values.astype(float)).float()
        pi_losses, v_losses, entropies = [], [], []
        if 'v_network' in self.config:
            for i in range(self.config.train_v_iter):  # update value function
                self.v_optimizer.zero_grad()
                target_values = torch.from_numpy(self.buffer.q_values.astype(float)).float()
                v_loss = self.compute_value_loss(observations, target_values, values)
                v_losses.append(v_loss.item())
                v_loss.backward()
                average_grads(MPI.COMM_WORLD, self.v_network.parameters())
                if self.id == 0:
                    self.v_optimizer.step()
                sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
            self.update_values(observations)
        # Update advantage estimates, standardizing across workers:
        self.estimate_advantages()
        advantages = torch.from_numpy(self.buffer.advantages.astype(float)).float()
        # Update policy network:
        actions = self.buffer.actions
        old_log_probs = torch.from_numpy(self.buffer.log_probs.astype(float)).float()
        if not self.config.trpo:
            for i in range(self.config.train_pi_iter):
                self.pi_optimizer.zero_grad()
                pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages, old_log_probs)
                if self.config.surrogate:  # can normalize this way to match safety-starter agents hyperparameters
                    pi_loss = torch.mean(pi_loss, dim=0)  # assumes equal experiences per worker
                else:
                    number_of_episodes = self.buffer.trajectories
                    pi_loss = torch.sum(pi_loss, dim=0) / number_of_episodes  # assumes equal episodes per worker
                pi_losses.append(pi_loss.item())
                entropies.append(entropy.item())
                mean_kld = mpi_avg(MPI.COMM_WORLD, kld)
                if mean_kld > self.config.max_kl > 0:
                    if self.id == 0:
                        print_now('Policy KL divergence exceeds limit; stopping update at step %d.' % i)
                    break
                pi_loss.backward()
                average_grads(MPI.COMM_WORLD, self.pi_network.parameters())
                if self.id == 0:
                    self.pi_optimizer.step()
                sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
            return {'pi_losses': pi_losses, 'v_losses': v_losses, 'entropies': entropies}
        else:  # TRPO update
            pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages, old_log_probs)
            pi_loss = torch.mean(pi_loss, dim=0)
            loss_current = mpi_avg(MPI.COMM_WORLD, pi_loss.item())
            pi_parameters = list(self.pi_network.parameters())
            loss_grad = self.flat_grad(pi_loss, pi_parameters, retain_graph=True)
            g = torch.from_numpy(mpi_avg(MPI.COMM_WORLD, loss_grad.data.numpy()))
            g_kl = self.flat_grad(kld, pi_parameters, create_graph=True)

            def hessian_vector_product(v):
                hvp = self.flat_grad(g_kl @ v, pi_parameters, retain_graph=True)
                hvp += self.config.damping_coeff * v
                return torch.from_numpy(mpi_avg(MPI.COMM_WORLD, hvp.data.numpy()))

            search_dir = self.conjugate_gradient(hessian_vector_product, g)
            max_length = torch.sqrt(2*self.config.max_kl/(search_dir @ hessian_vector_product(search_dir) + 1.e-8))
            max_step = max_length * search_dir

            def backtracking_line_search():

                def apply_update(grad_flattened):
                    n = 0
                    for p in pi_parameters:
                        numel = p.numel()
                        gf = grad_flattened[n:n + numel].view(p.shape)
                        p.data -= gf
                        n += numel

                loss_improvement = 0
                for r in self.config.backtrack_ratios:
                    step = r * max_step
                    apply_update(step)
                    with torch.no_grad():
                        loss_new, _, kld_new = self.compute_policy_loss(observations, actions, advantages,
                                                                        old_log_probs)
                        loss_new = mpi_avg(MPI.COMM_WORLD, loss_new.mean().item())
                        kld_new = mpi_avg(MPI.COMM_WORLD, kld_new.item())
                    loss_improvement = loss_current - loss_new
                    if loss_improvement > 0 and kld_new <= self.config.max_kl:
                        break
                    apply_update(-step)
                if loss_improvement <= 0 or kld_new > self.config.max_kl:
                    if loss_improvement <= 0:
                        print_zero(MPI.COMM_WORLD, 'step rejected; loss does not improve')
                    if kld_new > self.config.max_kl:
                        print_zero(MPI.COMM_WORLD, 'step rejected; max kld exceeded')
                    return loss_current
                else:
                    return loss_new

            final_loss = backtracking_line_search()
            return {'pi_losses': [final_loss], 'v_losses': v_losses, 'entropies': [entropy.item()]}

    def compute_value_loss(self, observations, target_values, old_values):
        """  Compute value function loss  """
        new_values = self.v_network(observations).view(-1)
        if self.config.v_clip > 0:
            clipped_values = old_values + torch.clamp(new_values - old_values,
                                                      -self.config.v_clip, self.config.v_clip)
            vf_losses_1 = torch.pow(new_values - target_values, 2)
            vf_losses_2 = torch.pow(clipped_values - target_values, 2)
            value_loss = torch.mean(torch.max(vf_losses_1, vf_losses_2), dim=0)
        else:
            value_loss = torch.mean(torch.pow(new_values - target_values, 2), dim=0)
        return value_loss

    def update_values(self, observations):
        """  Estimate values with updated value network, store in buffer  """
        with torch.no_grad():
            self.buffer.values = self.v_network(observations).view(-1).numpy()  # reward values
            # compute bootstraps:
            torch_bootstrap_obs = torch.from_numpy(self.buffer.bootstrap_obs).float()
            bootstrap_values = self.v_network(torch_bootstrap_obs).squeeze(-1).numpy()
            terminals = np.nonzero(self.buffer.dones.astype(int))[0]
            self.buffer.bootstraps = bootstrap_values * (terminals > 1)

    def estimate_advantages(self):
        """  Estimate advantages for a sequence of observations and rewards  """
        if not self.config.reward_to_go:
            self.buffer.advantages = self.buffer.q_values
        else:
            if 'v_network' in self.config:
                if self.config.gae:
                    rewards, values, dones = self.buffer.rewards, self.buffer.values, self.buffer.dones
                    self.buffer.advantages = self.estimate_generalized_advantage(rewards, values, dones)
                else:
                    self.buffer.advantages = self.buffer.q_values - self.buffer.values
            else:
                self.buffer.advantages = deepcopy(self.buffer.q_values)
        mean_adv, std_adv = mpi_statistics_scalar(MPI.COMM_WORLD, self.buffer.advantages)
        self.buffer.advantages = (self.buffer.advantages - mean_adv) / std_adv
        return mean_adv, std_adv

    def estimate_generalized_advantage(self, rewards, values, dones):
        """  Generalized advantage estimation, given rewards and value estimates for a given episode  """
        terminals = np.nonzero(dones.astype(int))[0]
        terminals = list(np.concatenate((np.array([-1]), terminals)))
        gae = np.zeros(rewards.shape)
        for i in range(len(terminals[:-1])):
            episode_rewards = rewards[terminals[i] + 1:terminals[i + 1] + 1]
            episode_values = values[terminals[i] + 1:terminals[i + 1] + 1]
            episode_next_values = np.concatenate((episode_values[1:], np.array([self.buffer.bootstraps[i]])))
            episode_deltas = episode_rewards + self.config.gamma * episode_next_values - episode_values
            for start in range(len(episode_values)):
                indices = np.arange(start, len(episode_values))
                discounts = np.power(self.config.gamma * self.config.lam, indices - start)
                discounted_future_deltas = episode_deltas[start:] * discounts
                gae[start + terminals[i] + 1] = np.sum(discounted_future_deltas)
        return gae

    def compute_policy_loss(self, observations, actions, advantages, old_log_probs, clip=True):
        """  Compute policy loss, entropy, kld  """
        new_policies = self.pi_network(observations)
        if self.pi_network.config.discrete:
            actions_one_hot = torch.from_numpy(
                np.eye(self.pi_network.config.action_dim)[np.squeeze(actions)]).float()
            new_policies = torch.masked_fill(new_policies, new_policies < self.epsilon, self.epsilon)
            new_log_probs = torch.sum(torch.log(new_policies) * actions_one_hot, dim=1)
        else:
            new_dist = self.sampler.get_distribution(new_policies)
            actions_torch = torch.from_numpy(np.vstack(actions)).float()
            if self.config.bound_corr:
                below = new_dist.cdf(actions_torch.clamp(-1.0, 1.0))
                below = below.clamp(self.epsilon, 1.0).log() * actions_torch.le(-1).float()
                above = torch.ones(actions_torch.size()) - new_dist.cdf(actions_torch.clamp(-1.0, 1.0))
                above = above.clamp(self.epsilon, 1.0).log() * actions_torch.ge(1).float()
                inner = new_dist.log_prob(actions_torch) * actions_torch.gt(-1).float() * actions_torch.lt(1).float()
                new_log_probs = torch.sum(inner + below + above, dim=-1)
            else:
                new_log_probs = torch.sum(new_dist.log_prob(actions_torch), dim=-1)
        entropy = self.sampler.compute_entropy(new_policies)
        if self.config.surrogate:
            ratio = torch.exp(new_log_probs - old_log_probs)
            if self.config.clip > 0 and clip:
                pi_losses_1 = -advantages * ratio
                pi_losses_2 = -advantages * torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip)
                pi_loss = torch.max(pi_losses_1, pi_losses_2)
            else:
                pi_loss = -advantages * ratio
        else:
            if self.config.clip > 0 and clip:
                log_diff = torch.clamp(new_log_probs - old_log_probs,
                                       np.log(1.0 - self.config.clip), np.log(1.0 + self.config.clip))
                pi_losses_1 = -advantages * new_log_probs
                pi_losses_2 = -advantages * (old_log_probs + log_diff)
                pi_loss = torch.max(pi_losses_1, pi_losses_2)
            else:
                pi_loss = -advantages * new_log_probs
        kld = self.compute_kld(new_policies, old_log_probs, new_log_probs)
        return pi_loss, entropy, kld

    def compute_kld(self, policy_predictions, old_log_probs, new_log_probs):
        """ Compute KL divergence for early stopping  """
        if self.config.trpo:
            new_dist = self.sampler.get_distribution(policy_predictions)
            old_dist = self.sampler.restore_distribution(self.buffer.policies)
            return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
        else:
            if self.config.full_kl:  # compute full kld
                kld = self.sampler.compute_kld(self.buffer.policies, policy_predictions)
            else:  # use approximation from Spinning Up
                kld = (old_log_probs - new_log_probs).mean().item()
            return kld

    def run_evaluation(self):
        """  Run evaluation episodes, collect data for logging  """
        # Run evaluation episodes:
        self.mode = 'test'
        self.obs, _ = self.reset_env()  # make sure environment is reset
        local_episodes, total_episodes, local_episode_summaries = 0, 0, {}
        self.pi_network.eval()
        if self.v_network is not None:
            self.v_network.eval()
        while total_episodes < self.config.evaluation_episodes:
            self.buffer = OnPolicyBuffer()  # reset experience buffer
            episode_summary = self.run_trajectory()
            local_episode_summaries = self.concatenate_dict_of_lists(local_episode_summaries, episode_summary)
            local_episodes += 1
            total_episodes = int(mpi_sum(MPI.COMM_WORLD, local_episodes))
        # Put back to resume training:
        self.mode = 'train'
        self.obs, _ = self.reset_env()
        # Collect, process, return data:
        episode_data = collect_dict_of_lists(MPI.COMM_WORLD, local_episode_summaries)
        evaluation_metrics = self.compute_metrics(episode_data)
        evaluation_info = self.process_evaluation_info(episode_data)
        return {**evaluation_metrics, **evaluation_info}

    def update_logging(self, episode_summaries, losses, evaluation, steps, previous_steps):
        """  Updates TensorBoard logging based on most recent update  """
        local_keys = list(episode_summaries.keys())
        all_keys = mpi_gather_objects(MPI.COMM_WORLD, local_keys)
        keys_in_each = self.find_common(all_keys)
        for k in keys_in_each:
            self.logger.log_mean_value('Performance/' + k, episode_summaries[k], steps, previous_steps)
        # for k, v in losses.items():
        #     self.logger.log_mean_value('Losses/' + k, v, steps, previous_steps)
        if evaluation is not None:
            for k, v in evaluation.items():
                self.logger.log_scalar('Evaluation/' + k, v, steps, previous_steps)
        self.logger.flush()

    def save_networks(self, total_steps, last_checkpoint):
        """  Save networks, as required.  Update last_checkpoint.  """
        self.pi_network.save(os.path.join(self.config.model_folder, 'model-latest.pt'))
        if self.v_network is not None:
            self.v_network.save(os.path.join(self.config.model_folder, 'value-latest.pt'))
        if total_steps // self.config.checkpoint_every > last_checkpoint:  # Periodically keep checkpoint
            last_checkpoint += 1
            suffix = str(int(last_checkpoint * self.config.checkpoint_every))
            self.pi_network.save(os.path.join(self.config.model_folder, 'model-' + suffix + '.pt'))
            if self.v_network is not None:
                self.v_network.save(os.path.join(self.config.model_folder, 'value-' + suffix + '.pt'))
        return last_checkpoint

    def test(self):
        """  Run testing episodes with fixed random seed, collect and save data  """
        # Run testing episodes:
        self.mode = 'test'
        self.initialize_env()
        self.initialize_networks()
        self.sampler = get_sampler(self.config.pi_network, True)
        self.obs, _ = self.reset_env()
        local_episodes, total_episodes, local_episode_summaries = 0, 0, {}
        self.pi_network.eval()
        if self.v_network is not None:
            self.v_network.eval()
        while total_episodes < self.config.test_episodes:
            self.buffer = OnPolicyBuffer()  # reset experience buffer
            random_seed = self.id*self.config.test_episodes + self.config.test_random_base + local_episodes + 1
            episode_summary = self.run_trajectory(int(random_seed))
            local_episode_summaries = self.concatenate_dict_of_lists(local_episode_summaries, episode_summary)
            local_episodes += 1
            total_episodes = int(mpi_sum(MPI.COMM_WORLD, local_episodes))
            if self.id == 0:
                print(str(total_episodes) + ' episodes complete.')
        # Collect, process, save data:
        test_output = collect_dict_of_lists(MPI.COMM_WORLD, local_episode_summaries)
        self.store_test_results(test_output)
        return test_output

    def store_test_results(self, test_output):
        """  Save a pickle with test results  """
        if self.id == 0:
            test_file = os.path.join(self.config.model_folder, 'test_results.pkl')
            with open(test_file, 'wb') as opened_test:
                pickle.dump(test_output, opened_test)

    def conjugate_gradient(self, Ax, b):
        x = torch.zeros_like(b)
        r = b.clone()  # residual
        p = b.clone()  # basis vector
        epsilon = 1.e-8*torch.ones((r@r).size())
        for _ in range(self.config.cg_iter):
            z = Ax(p)
            r_dot_old = r @ r
            alpha = r_dot_old / ((p @ z) + epsilon)
            x_new = x + alpha * p
            if (x - x_new).norm() <= self.config.cg_delta:
                return x_new
            r = r - alpha * z
            beta = (r @ r) / r_dot_old
            p = r + beta * p
            x = x_new
        return x

    def step_env(self, action, truncated):
        """ Steps input environment, accommodating Gym, Gymnasium APIs"""
        step_output = self.env.step(action)
        if len(step_output) == 4:  # gym
            next_obs, reward, terminated, info = step_output
        else:  # gymnasium
            next_obs, reward, terminated, truncated, info = step_output
        if truncated:
            terminated = False
        return next_obs, reward, terminated, truncated, info

    def reset_env(self):
        """  Resets an environment, accommodating Gym, Gymnasium APIs """
        outputs = self.env.reset()
        if isinstance(outputs, tuple):
            return outputs
        else:
            return outputs, {}

    @staticmethod
    def flat_grad(y, x, retain_graph=False, create_graph=False):
        """  Compute a flat version of gradient of y wrt x  """
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    @staticmethod
    def compute_metrics(episode_data):
        """  Computes metrics to be evaluated as learning progresses  """
        mean_reward = sum(episode_data['episode_reward']) / len(episode_data)
        return {'mean': mean_reward}

    @staticmethod
    def process_evaluation_info(episode_data):
        """  Processes evaluation info, returning dictionary of mean values """
        mean_info = {}
        for k, v in episode_data.items():
            if k[:5] == 'info_':
                mean_info[k] = sum(v) / len(v)
        return mean_info

    @staticmethod
    def concatenate_dict_of_arrays(base_dict, new_dict):
        """  Collect a dictionary of numpy arrays  """
        for k in new_dict:
            base_dict[k] = np.concatenate((base_dict[k], new_dict[k]))
        return base_dict

    @staticmethod
    def concatenate_dict_of_lists(base_dict, new_dict):
        """  Collect a dictionary of lists  """
        if len(base_dict.keys()) == 0:  # first iteration
            base_dict = {k: [v] for k, v in new_dict.items()}
        else:
            for k in new_dict:
                if k in base_dict:
                    base_dict[k].append(new_dict[k])
                else:
                    base_dict[k] = [new_dict[k]]
        return base_dict

    @staticmethod
    def find_common(list_of_lists):
        """  Returns members common to each list in a list of lists  """
        common = set(list_of_lists[0])
        for item in list_of_lists[1:]:
            common = common.intersection(set(item))
        return sorted(list(common))


if __name__ == '__main__':
    """  Runs PolicyOptimizer training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--trpo', help='whether to force trpo update', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = Box(json.load(f1))
    config1.seed = in_args.seed
    if 'trpo' not in config1:
        config1.trpo = bool(in_args.trpo)
    if in_args.mode.lower() == 'train':
        pg_object = PolicyOptimizer(config1)
        pg_object.train()
    else:
        config1.use_prior_nets = True
        pg_object = PolicyOptimizer(config1)
        pg_object.test()
