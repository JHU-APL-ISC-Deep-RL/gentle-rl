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
from warnings import warn
from gentle.common.loggers import Logger
from gentle.common.buffers import OnPolicyBuffer, OffPolicyBuffer
from gentle.common.utils import get_env_object, get_network_object, get_sampler
from gentle.common.mpi_data_utils import mpi_sum, average_grads, sync_weights, mpi_statistics_scalar, \
    mpi_avg, collect_dict_of_lists, mpi_gather_objects, print_zero


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class ConservativeSafetyCritic(object):

    def __init__(self, config):
        """  Conservative Safety Critic for Exploration, parallelized with MPI.  Can use either TRPO or PPO.  """
        self.id = MPI.COMM_WORLD.Get_rank()
        self.config = config
        self.mode = ''
        self.env, self.eval_env = None, None
        self.obs = None
        self.pi_network, self.pi_optimizer = None, None
        self.v_network, self.v_optimizer = None, None
        self.qc_network, self.qc_optimizer = None, None
        self.qc_target = None
        self.pi_buffer, self.qc_buffer, self.mb = None, None, None
        self.penalty_param, self.penalty_optimizer = None, None
        self.penalty = 0
        self.sampler = None
        self.logger = None
        self.epsilon = 1.e-6
        self.num_workers = MPI.COMM_WORLD.Get_size()
        self.process_config()
        torch.manual_seed((self.id + 1 + self.config.seed * self.num_workers) * 2000)
        np.random.seed((self.id + 1 + self.config.seed * self.num_workers) * 5000)

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate  """
        # General training configuration:
        self.config.setdefault('seed', 0)  # Random seed parameter
        self.config.setdefault('training_frames', int(5e7))  # Number of frames to use for training run
        self.config.setdefault('max_ep_length', -1)  # Maximum episode length (< 0 means no max)
        self.config.setdefault('batch_size', 4000)  # Average number of experiences to base an update on
        self.config.setdefault('pi_lr', .0003)  # Policy optimizer learning rate
        self.config.setdefault('v_lr', .001)  # Value optimizer learning rate
        self.config.setdefault('q_lr', 0.0002)  # Qc optimizer learning rate
        self.config.setdefault('train_v_iter', 80)  # Value updates per epoch
        self.config.setdefault('train_q_iter', 80)  # Q updates per epoch
        self.config.setdefault('train_pi_iter', 80)  # Policy updates per epoch (for PPO)
        self.config.setdefault('gamma', 0.99)  # Discount factor gamma
        self.config.setdefault('lam', 0.95)  # GAE factor lambda
        self.config.setdefault('clip', 0.2)  # Clip factor for policy (PPO only; < 0 means none)
        self.config.setdefault('max_kl', -1)  # KL criteria for early stopping (< 0 means ignore)
        self.config.setdefault('full_gae', True)  # whether to put cost into GAE
        self.config.setdefault('minibatch_size', -1)  # minibatch size; -1 means to use full buffer
        # Q-learning, action selection:
        self.config.setdefault('q_batch_size', 256)  # one worker handles Q update
        self.config.setdefault('alpha', 5)  # CQL weight
        self.config.setdefault('polyak', 0.995)  # weight of previous network weights in Polyak averaging
        self.config.setdefault('buffer_size', 1e6)  # capacity of replay buffer
        self.config.setdefault('num_actions', 100)  # choose num_actions > 1 to activate safety critic
        self.config.setdefault('num_cql', 100)  # num actions sampled for conservative Q learning
        # TRPO-specific configuration:
        self.config.setdefault('trpo', True)
        if self.config.trpo:
            self.config.train_pi_iter = 1  # Policy updates per epoch
            self.config.clip = -1  # Clip factor for policy (< 0 means none)
            self.config.setdefault('cg_iter', 10)  # Number of iterations in conjugate gradient method
            self.config.setdefault('cg_delta', 0)  # Early stopping in conjugate gradient solver
            self.config.setdefault('damping_coeff', 0.1)  # Improves numerical stability of hessian vector product
            self.config.setdefault('backtrack_iter', 10)  # Maximum number of backtracks allowed per line search
            self.config.setdefault('backtrack_coeff', 0.8)  # How far back to step during backtracking line search
            self.config.backtrack_ratios = self.config.backtrack_coeff ** np.arange(self.config.backtrack_iter)
        # Variance reduction measures:
        self.config.setdefault('full_kl', True)  # Whether to use full KL estimation or approximation
        self.config.setdefault('bound_corr', False)  # Whether to use boundary correction in policy log-prob
        self.config.pi_network.bound_corr = self.config.bound_corr
        # Testing configuration:
        self.config.setdefault('test_episodes', 1000)
        self.config.setdefault('render', False)
        # Logging and storage configurations:
        self.config.setdefault('checkpoint_every', int(1e7))
        self.config.setdefault('evaluation_every', -1)  # By default, don't run evaluations
        self.config.setdefault('evaluation_episodes', 1)
        self.config.setdefault('enable_restart', False)  # Store optimizers, target, buffer for restart
        self.config.setdefault('use_prior_nets', False)  # Whether to pick up where previous training left off
        self.config.setdefault('model_folder', '../../output/csc')
        self.config.setdefault('log_folder', '../../logs/csc')
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
        # Process configuration related to constraint:
        self.config.setdefault('cost_limit', 25)
        self.config.setdefault('penalty_init', .025)                    # initial value of penalty constant
        self.config.setdefault('penalty_lr', .0001)                     # learning rate for Lagrange multiplier

    def train(self):
        """  Train neural network  """
        # Initialize relevant objects:
        self.mode = 'train'
        self.initialize_env()
        self.initialize_networks()
        self.sampler = get_sampler(self.config.pi_network)
        self.initialize_qc_buffer()
        total_steps, last_checkpoint, last_evaluation = self.initialize_optimizers()
        self.initialize_logging()
        self.allocate_minibatches()
        # Run training:
        total_steps = max(last_checkpoint * self.config.checkpoint_every, 0)
        while total_steps < self.config.training_frames:
            # Collect data, update networks:
            self.collect_data()
            total_steps += self.config.batch_size * self.num_workers
            # Update networks:
            loss_summary = self.update_networks()
            # Evaluate, log, save the model:
            if total_steps // self.config.evaluation_every > last_evaluation:
                evaluation_summary = self.run_evaluation()
                last_evaluation += 1
            else:
                evaluation_summary = None
            self.update_logging(loss_summary, evaluation_summary, total_steps)
            if self.id == 0:
                last_checkpoint = self.save_training(total_steps, last_checkpoint)

    def collect_data(self):
        """  Collect data for training  """
        self.pi_buffer = OnPolicyBuffer()  # reset on-policy experience buffer
        current_buffer = np.array([]).reshape(0, 8)
        batch_frames = 0
        while True:
            policy, value = self.forward_pass(self.obs)
            action, log_prob = self.get_action_and_log_prob(policy)
            trunc = self.pi_buffer.episode_lengths[-1] == self.config.max_ep_length - 1
            output_obs, reward, terminated, truncated, info = self.step_env(action, trunc)
            cost = info.get('cost', 0)
            batch_frames += 1
            self.pi_buffer.update_episode(reward, cost)
            current_buffer = self.update_current_buffer(current_buffer, action, reward, policy,
                                                        log_prob, value, terminated + 2 * truncated, cost)
            self.obs = output_obs
            if terminated or truncated or batch_frames == self.config.batch_size:
                self.pi_buffer.log_bootstrap_obs(self.obs)
                self.process_buffer(current_buffer, truncated)
                if terminated or truncated:
                    self.obs, _ = self.reset_env()
                    self.pi_buffer.reset_episode()
                    current_buffer = np.array([]).reshape(0, 8)
                if batch_frames == self.config.batch_size:
                    break
        self.collect_qc_buffers()

    def initialize_env(self):
        """  Initialize environment object.  """
        self.env = get_env_object(self.config)
        self.obs, _ = self.reset_env()
        if self.config.evaluation_every > 0 or self.mode == 'test':
            self.eval_env = get_env_object(self.config)

    def initialize_qc_buffer(self):
        """  Initialize replay buffer  """
        self.qc_buffer = OffPolicyBuffer(capacity=self.config.buffer_size, obs_dim=self.env.observation_space.shape[0],
                                         act_dim=self.env.action_space.shape[0], store_costs=True)
        if self.config.use_prior_nets and self.id == 1:
            self.qc_buffer.load(os.path.join(self.config.model_folder, 'buffer-latest.p.tar.gz'))

    def initialize_networks(self):
        """  Initialize network objects  """
        self.pi_network = get_network_object(self.config.pi_network)
        self.v_network = get_network_object(self.config.v_network)
        self.qc_network = get_network_object(self.config.q_network)
        self.qc_target = deepcopy(self.qc_network)
        self.penalty_param = torch.autograd.Variable(torch.ones(1) * self.config.penalty_init, requires_grad=True)
        if self.config.use_prior_nets:
            self.pi_network.restore(os.path.join(self.config.model_folder, 'model-latest.pt'))
            self.v_network.restore(os.path.join(self.config.model_folder, 'value-latest.pt'))
            self.qc_network.restore(os.path.join(self.config.model_folder, 'qc-latest.pt'))
            self.qc_target.restore(os.path.join(self.config.model_folder, 'qc-target.pt'))
            self.penalty_param = torch.load(os.path.join(self.config.model_folder, 'penalty-latest' + '.pt'))
        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.qc_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.qc_target.parameters())
        sync_weights(MPI.COMM_WORLD, self.penalty_param)

    def initialize_optimizers(self):
        """  Initializes Adam optimizer for training network.  Only one worker actually updates parameters.  """
        self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.pi_lr)
        self.v_optimizer = torch.optim.Adam(params=self.v_network.parameters(), lr=self.config.v_lr)
        if self.id == 0:
            self.qc_optimizer = torch.optim.Adam(params=self.qc_network.parameters(), lr=self.config.q_lr)
            self.penalty_optimizer = torch.optim.Adam(params=[self.penalty_param], lr=self.config.penalty_lr)
        total_steps, last_checkpoint, last_evaluation = 0, -1, -1
        if self.config.use_prior_nets:
            checkpoint = torch.load(os.path.join(self.config.model_folder, 'chkpt-latest.pkl'))
            self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
            self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
            if self.id == 0:
                self.qc_optimizer.load_state_dict(checkpoint['qc_optimizer'])
                self.penalty_optimizer.load_state_dict(checkpoint['penalty_optimizer'])
            total_steps = checkpoint['steps']
            last_checkpoint = total_steps // self.config.checkpoint_every
            last_evaluation = total_steps // self.config.evaluation_every
        return total_steps, last_checkpoint, last_evaluation

    def initialize_logging(self):
        """  Initialize logger and store config (only on one process)  """
        if self.id == 0:
            self.logger = Logger(self.config.log_folder)
            if not self.config.use_prior_nets:
                with open(os.path.join(self.config.model_folder, 'config.pkl'), 'wb') as config_file:
                    pickle.dump(self.config, config_file)  # store configuration
                self.logger.log_config(self.config)
                self.logger.log_graph(self.obs, self.pi_network)

    def allocate_minibatches(self):
        """  Return minibatch indices  """
        indices = list(range(self.config.batch_size))
        if self.config.minibatch_size > 0:
            starts = list(range(0, len(indices), self.config.minibatch_size))
            stops = [item + self.config.minibatch_size for item in starts]
            if stops[-1] != len(indices):
                warn('Trajectory length is not a multiple of minibatch size; wasting data')
            stops[-1] = len(indices)
        else:
            starts, stops = [0], [len(indices)]
        self.mb = Box({'indices': indices, 'starts': starts, 'stops': stops})
        return indices, starts, stops

    def forward_pass(self, obs):
        """  Runs forward pass of network(s).  For continuous action spaces, policy will be a tuple of mean, std. """
        policy = self.pi_network.forward_with_processing(obs)
        value = self.v_network.forward_with_processing(obs)
        value = value.detach().numpy()[0]
        return policy, value

    def get_action_and_log_prob(self, pi):
        """  Get action, optionally integrating a safety critic  """
        with torch.no_grad():
            if self.config.num_actions == 1:
                return self.sampler.get_action_and_log_prob(pi)
            else:  # use safety critic to choose action with minimum Qc
                obs_torch = torch.from_numpy(self.obs).float()
                obs_repeated = obs_torch[None, ...].repeat(self.config.num_actions, 1)
                pi = self.pi_network(obs_repeated)
                act, log_prob = self.sampler.get_action_and_log_prob(pi)
                act = torch.from_numpy(act).float()
                qc_act = self.qc_network(torch.cat((obs_repeated, act), dim=-1)).squeeze(-1)
                return act[torch.argmin(qc_act)].numpy(), log_prob[torch.argmin(qc_act)]

    def update_current_buffer(self, current_buffer, action, reward, policy, log_prob, value, done, cost=None):
        """  Updates episode buffer for current step  """
        policy_to_store = np.concatenate((policy[0].detach().numpy(), policy[1].detach().numpy()))
        experience = np.reshape(np.array([self.obs, action, reward, policy_to_store, log_prob, value, done, cost],
                                         dtype=object), (1, 8))
        return np.concatenate((current_buffer, experience))

    def process_buffer(self, current_buffer, truncated=False):
        """  Processes a completed episode, storing required data in buffer  """
        if self.mode == 'train':
            q_values = self.compute_target_values(current_buffer[:, 2])
            c_q_values = self.compute_target_values(current_buffer[:, 7])
            self.pi_buffer.update(current_buffer, q_values=q_values, c_q_values=c_q_values)
            if truncated:
                current_buffer[-1, 6] = False  # don't count truncated as done
            self.qc_buffer.store_raw(current_buffer, self.obs)

    def collect_qc_buffers(self):
        all_buffers = mpi_gather_objects(MPI.COMM_WORLD, self.qc_buffer.raw_data)
        if self.id == 0:  # update Q using only one worker
            for i in range(len(all_buffers)):
                for j in range(len(all_buffers[i])):
                    self.qc_buffer.trajectory_update(*all_buffers[i][j])
        self.qc_buffer.raw_data = []

    def compute_target_values(self, rewards):
        """  Computes value function targets (without bootstrapping)  """
        trajectory_length = rewards.shape[0]
        target_values = np.zeros((trajectory_length,))
        for start in range(trajectory_length):
            indices = np.arange(start, trajectory_length)
            discounts = np.power(self.config.gamma, indices - start)
            discounted_future_rewards = rewards[start:] * discounts
            target_values[start] = np.sum(discounted_future_rewards)
        return target_values

    def update_networks(self):
        """  Updates the networks based on processing from all workers  """
        # Update Lagrange multiplier:
        self.update_penalty()
        # Update value network:
        observations = torch.from_numpy(np.vstack(self.pi_buffer.observations)).float()
        pi_losses, v_losses, entropies = [], [], []
        # Update reward value function:
        for i in range(self.config.train_v_iter):
            np.random.shuffle(self.mb.indices)
            for start, stop in zip(self.mb.starts, self.mb.stops):
                mb = self.mb.indices[start:stop]
                self.v_optimizer.zero_grad()
                target_values = torch.from_numpy(self.pi_buffer.q_values[mb].astype(float)).float()
                v_loss = self.compute_value_loss(observations[mb], target_values[mb])
                v_losses.append(v_loss.item())
                v_loss.backward()
                average_grads(MPI.COMM_WORLD, self.v_network.parameters())
                if self.id == 0:
                    self.v_optimizer.step()
                sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
        # Update cost Q function:
        for i in range(self.config.train_q_iter):
            if self.id == 0:  # update Qc using only first process
                self.qc_optimizer.zero_grad()
                qc_loss = self.compute_qc_loss()
                qc_loss.backward()
                self.qc_optimizer.step()
            sync_weights(MPI.COMM_WORLD, self.qc_network.parameters())
            self.update_qc_target()
        # Update advantage estimates, standardizing across workers (includes conservatism):
        self.update_values(observations)
        self.estimate_advantages()
        advantages = torch.from_numpy(self.pi_buffer.advantages.astype(float)).float()
        # Update policy network:
        actions = self.pi_buffer.actions
        old_log_probs = torch.from_numpy(self.pi_buffer.log_probs.astype(float)).float()
        if not self.config.trpo:
            kld_flag = False
            for i in range(self.config.train_pi_iter):
                np.random.shuffle(self.mb.indices)
                for start, stop in zip(self.mb.starts, self.mb.stops):
                    mb = self.mb.indices[start:stop]
                    self.pi_optimizer.zero_grad()
                    pi_loss, entropy, kld = self.compute_policy_loss(observations[mb], actions[mb],
                                                                     advantages[mb], old_log_probs[mb])
                    pi_loss = torch.mean(pi_loss, dim=0)
                    pi_losses.append(pi_loss.item())
                    entropies.append(entropy.item())
                    mean_kld = mpi_avg(MPI.COMM_WORLD, kld)
                    if mean_kld > self.config.max_kl > 0:
                        kld_flag = True
                        if self.id == 0:
                            print('Policy KL divergence exceeds limit; stopping update at step %d.' % i, flush=True)
                        break
                    pi_loss.backward()
                    average_grads(MPI.COMM_WORLD, self.pi_network.parameters())
                    if self.id == 0:
                        self.pi_optimizer.step()
                    sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
                if kld_flag:
                    break
            return {'pi_losses': pi_losses, 'v_losses': v_losses,
                    'entropies': entropies, 'penalty': [self.penalty]}
        else:  # TRPO update; assuming we won't want minibatches for it
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
            return {'pi_losses': [final_loss], 'v_losses': v_losses,
                    'entropies': [entropy.item()], 'penalty': [self.penalty]}

    def update_penalty(self):
        """  Updates Lagrange multiplier based on current batch of data  """
        mean_cost, _ = mpi_statistics_scalar(MPI.COMM_WORLD, self.pi_buffer.episode_costs)
        if self.id == 0:
            self.penalty_optimizer.zero_grad()
            penalty_loss = (self.config.cost_limit - mean_cost) * self.penalty_param
            penalty_loss.backward()
            self.penalty_optimizer.step()
        sync_weights(MPI.COMM_WORLD, self.penalty_param)
        self.penalty = torch.nn.functional.relu(self.penalty_param).item()

    def compute_value_loss(self, observations, target_values):
        """  Compute value function loss  """
        new_values = self.v_network(observations).view(-1)
        return torch.mean(torch.pow(new_values - target_values, 2), dim=0)

    def compute_qc_loss(self):
        """  Conservative Qc update (including Bellman, CQL contributions)  """
        data = self.qc_buffer.sample(self.config.q_batch_size)
        # Bellman contribution:
        qc = self.qc_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        with torch.no_grad():  # just computing targets
            pi_next = self.pi_network(data.next_obs)
            act_next, log_prob_pi_next = self.sampler.get_action_and_log_prob(pi_next)
            act_next = torch.from_numpy(act_next).float()
            qc_target = self.qc_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
            not_dones = (torch.ones_like(data.terminated) - data.terminated).squeeze(-1)
            backup = data.rewards.squeeze(-1) + self.config.gamma * not_dones * qc_target  # currently, no max entropy
        bellman_loss = ((qc - backup) ** 2).mean()
        # CQL contribution:
        cql_loss = self.compute_cql_loss(data, qc)
        # Combination:
        qc_loss = bellman_loss + cql_loss
        return qc_loss

    def compute_cql_loss(self, data, q_data):
        """  Returns CQL contribution to Qc loss; maximizes Qc from new policy, minimizes from data  """
        repeated_current_obs = self.repeat_obs(data.obs, self.config.num_cql)  # [B*R, O]
        # current obs, current actions:
        pi_current = self.pi_network(repeated_current_obs)  # [B*R, A], [B*R, A]
        act_current, _ = self.sampler.get_action_and_log_prob(pi_current)  # [B*R, A], [B*R]
        act_current = torch.from_numpy(act_current).float()
        qc_current = self.qc_network(torch.cat((repeated_current_obs, act_current),
                                               dim=-1)).view(self.config.q_batch_size, -1)  # [B, R]
        cql_loss = self.config.alpha * (q_data - qc_current.mean(dim=-1)).mean()
        return cql_loss

    def update_qc_target(self):
        """  Update target networks via Polyak averaging  """
        with torch.no_grad():
            for p, p_targ in zip(self.qc_target.parameters(), self.qc_target.parameters()):
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)

    def update_values(self, observations):
        """  Update values in buffer, including bootstraps  """
        with torch.no_grad():
            self.pi_buffer.values = self.v_network(observations).view(-1).numpy()  # reward values
            qc_data_in = torch.cat((observations, torch.from_numpy(self.pi_buffer.actions).float()), dim=-1)
            self.pi_buffer.cost_values = self.qc_network(qc_data_in).squeeze(-1).numpy()  # cost values
            # compute bootstraps:
            torch_bootstrap_obs = torch.from_numpy(self.pi_buffer.bootstrap_obs).float()
            bootstrap_values = self.v_network(torch_bootstrap_obs).squeeze(-1).numpy()
            if self.config.full_gae:
                policy = self.pi_network(torch_bootstrap_obs)
                actions, _ = self.sampler.get_action_and_log_prob(policy)
                qc_input = torch.cat((torch_bootstrap_obs, torch.from_numpy(actions).float()), dim=-1)
                c_values = self.qc_network(qc_input).squeeze(-1).numpy()
                bootstrap_values -= self.penalty * c_values
            terminals = self.pi_buffer.dones[np.nonzero(self.pi_buffer.dones.astype(int))[0]].astype(int)
            self.pi_buffer.bootstraps = bootstrap_values * (terminals > 1)

    def estimate_advantages(self):
        """  Estimate advantages for a sequence of observations and rewards  """
        if self.config.full_gae:
            rewards = self.pi_buffer.rewards.astype(float) - self.penalty * self.pi_buffer.costs.astype(float)
            values = self.pi_buffer.values - self.penalty * self.pi_buffer.cost_values
            dones = self.pi_buffer.dones
            self.pi_buffer.advantages = self.estimate_generalized_advantage(rewards, values, dones)
        else:
            with torch.no_grad():  # conservative cost advantage estimation
                torch_obs = torch.from_numpy(self.pi_buffer.log_bootstrap_obs).float()
                pi_data = self.pi_network(torch_obs)
                act_curr, _ = self.sampler.get_action_and_log_prob(pi_data)
                qc_curr_in = torch.cat((torch_obs, torch.from_numpy(act_curr).float()), dim=-1)
                qc_curr = self.qc_network(qc_curr_in).squeeze(-1).numpy()  # value estimate
                cost_adv = self.penalty * (self.pi_buffer.cost_values - qc_curr)  # absorb (1-gamma) into self.penalty
            rewards = self.pi_buffer.rewards
            values = self.pi_buffer.values
            dones = self.pi_buffer.dones
            reward_adv = self.estimate_generalized_advantage(rewards, values, dones)
            self.pi_buffer.advantages = reward_adv - cost_adv
        mean_adv, std_adv = mpi_statistics_scalar(MPI.COMM_WORLD, self.pi_buffer.advantages)
        self.pi_buffer.advantages = (self.pi_buffer.advantages - mean_adv) / std_adv
        return mean_adv, std_adv

    def estimate_generalized_advantage(self, rewards, values, dones):
        """  Generalized advantage estimation, given rewards and value estimates for a given episode  """
        terminals = np.nonzero(dones.astype(int))[0]
        terminals = list(np.concatenate((np.array([-1]), terminals)))
        gae = np.zeros(rewards.shape)
        for i in range(len(terminals[:-1])):
            episode_rewards = rewards[terminals[i] + 1:terminals[i + 1] + 1]
            episode_values = values[terminals[i] + 1:terminals[i + 1] + 1]
            episode_next_values = np.concatenate((episode_values[1:], np.array([self.pi_buffer.bootstraps[i]])))
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
        ratio = torch.exp(new_log_probs - old_log_probs)
        if self.config.clip > 0 and clip:
            pi_losses_1 = -advantages * ratio
            pi_losses_2 = -advantages * torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip)
            pi_loss = torch.max(pi_losses_1, pi_losses_2)
        else:
            pi_loss = -advantages * ratio
        kld = self.compute_kld(new_policies, old_log_probs, new_log_probs)
        return pi_loss, entropy, kld

    def compute_kld(self, policy_predictions, old_log_probs, new_log_probs):
        """ Compute KL divergence for early stopping  """
        if self.config.trpo:
            new_dist = self.sampler.get_distribution(policy_predictions)
            old_dist = self.sampler.restore_distribution(self.pi_buffer.policies)
            return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
        else:
            if self.config.full_kl:  # compute full kld
                kld = self.sampler.compute_kld(self.pi_buffer.policies, policy_predictions)
            else:  # use approximation from Spinning Up
                kld = (old_log_probs - new_log_probs).mean().item()
            return kld

    def run_evaluation(self):
        """  Run evaluation episodes, collect data for logging  """
        # Run evaluation episodes:
        local_episodes, total_episodes, local_episode_summaries = 0, 0, {}
        while total_episodes < self.config.evaluation_episodes:
            episode_summary = self.run_trajectory()
            local_episode_summaries = self.concatenate_dict_of_lists(local_episode_summaries, episode_summary)
            local_episodes += 1
            total_episodes = int(mpi_sum(MPI.COMM_WORLD, local_episodes))
        # Collect, process, return data:
        episode_data = collect_dict_of_lists(MPI.COMM_WORLD, local_episode_summaries)
        evaluation_metrics = self.compute_metrics(episode_data)
        return {**episode_data, **evaluation_metrics}

    def test(self):
        """  Run testing episodes with fixed random seed, collect and save data  """
        # Run testing episodes:
        self.mode = 'test'
        self.initialize_env()
        self.initialize_networks()
        self.sampler = get_sampler(self.config.pi_network, True)
        self.obs, _ = self.reset_env()
        self.config.evaluation_episodes = self.config.test_episodes
        test_output = self.run_evaluation()
        self.store_test_results(test_output)

    def run_trajectory(self):
        """  Run trajectories for testing based on current network(s)  """
        episode_reward, episode_cost, episode_length = 0, 0, []
        obs = self.reset_env(train=False)
        while True:
            policy, value = self.forward_pass(obs)
            action, log_prob = self.get_action_and_log_prob(policy)
            trunc = episode_length == self.config.max_ep_length - 1
            obs, reward, terminated, truncated, info = self.step_env(action, trunc, train=False)
            cost = info.get('cost', 0)
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            if self.config.render:
                self.env.render()
            if terminated or truncated:
                break
        return {'episode_reward': episode_reward, 'episode_cost': episode_cost, 'episode_length': episode_length}

    def update_logging(self, loss_info, evaluation_info, steps):
        """  Updates TensorBoard logging based on most recent update  """
        rewards = mpi_gather_objects(MPI.COMM_WORLD, self.pi_buffer.episode_rewards)
        costs = mpi_gather_objects(MPI.COMM_WORLD, self.pi_buffer.episode_costs)
        lengths = mpi_gather_objects(MPI.COMM_WORLD, self.pi_buffer.episode_lengths)
        if self.id == 0:
            for k, v in loss_info.items():
                self.logger.log_mean_value('Learning/' + k, v, steps)
            if evaluation_info is not None:
                for k, v in evaluation_info.items():
                    self.logger.log_mean_value('Evaluation/' + k, v, steps)
            rewards_to_log = self.flatten_list([item[:-1] for item in rewards])
            costs_to_log = self.flatten_list([item[:-1] for item in costs])
            lengths_to_log = self.flatten_list([item[:-1] for item in lengths])
            self.logger.log_mean_value('Train/rewards', rewards_to_log, steps)
            self.logger.log_mean_value('Train/costs', costs_to_log, steps)
            self.logger.log_mean_value('Train/lengths', lengths_to_log, steps)
            self.logger.flush()
        self.pi_buffer.reset_logging()

    def save_training(self, total_steps, last_checkpoint):
        """  Save networks, as required.  Update last_checkpoint.  """
        self.pi_network.save(os.path.join(self.config.model_folder, 'model-latest.pt'))
        self.v_network.save(os.path.join(self.config.model_folder, 'value-latest.pt'))
        self.qc_network.save(os.path.join(self.config.model_folder, 'qc-latest.pt'))
        torch.save(self.penalty_param, os.path.join(self.config.model_folder, 'penalty-latest.pt'))
        if self.config.enable_restart:
            self.qc_target.save(os.path.join(self.config.model_folder, 'qc-target.pt'))
            torch.save({'steps': total_steps,
                        'pi_optimizer': self.pi_optimizer.state_dict(),
                        'v_optimizer': self.v_optimizer.state_dict(),
                        'qc_optimizer': self.qc_optimizer.state_dict(),
                        'penalty_optimizer': self.penalty_optimizer.state_dict()},
                       os.path.join(self.config.model_folder, 'chkpt-latest.pt'))
            self.qc_buffer.save(os.path.join(self.config.model_folder, 'buffer-latest.p.tar.gz'))
        if total_steps // self.config.checkpoint_every > last_checkpoint:  # Periodically keep checkpoint
            last_checkpoint += 1
            suffix = str(int(last_checkpoint * self.config.checkpoint_every))
            self.pi_network.save(os.path.join(self.config.model_folder, 'model-' + suffix + '.pt'))
            self.v_network.save(os.path.join(self.config.model_folder, 'value-' + suffix + '.pt'))
            self.qc_network.save(os.path.join(self.config.model_folder, 'qc-' + suffix + '.pt'))
            torch.save(self.penalty_param, os.path.join(self.config.model_folder, 'penalty-' + suffix + '.pt'))
        return last_checkpoint

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

    def step_env(self, action, truncated, train=True):
        """ Steps input environment, accommodating Gym, Gymnasium APIs"""
        if train:
            step_output = self.env.step(action)
        else:
            step_output = self.eval_env.step(action)
        if len(step_output) == 4:  # gym
            next_obs, reward, terminated, info = step_output
        else:  # gymnasium
            next_obs, reward, terminated, truncated, info = step_output
        if truncated:
            terminated = False
        return next_obs, reward, terminated, truncated, info

    def reset_env(self, train=True):
        """  Resets an environment, accommodating Gym, Gymnasium APIs """
        if train:
            outputs = self.env.reset()
        else:
            outputs = self.eval_env.reset()
        if isinstance(outputs, tuple):
            return outputs
        else:
            return outputs, {}

    def compute_metrics(self, episode_data):
        """  Computes metrics to be evaluated as learning progresses  """
        mean_total = (sum(episode_data['episode_reward']) - self.penalty * sum(episode_data['episode_cost']))\
            / len(episode_data)
        return {'mean': mean_total}

    @staticmethod
    def repeat_obs(obs, num_repeat):
        num_obs = obs.shape[0]  # B
        repeated_obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(num_obs * num_repeat, -1)  # [B*R, O]
        return repeated_obs

    @staticmethod
    def flat_grad(y, x, retain_graph=False, create_graph=False):
        """  Compute a flat version of gradient of y wrt x  """
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

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
    def flatten_list(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]


if __name__ == '__main__':
    """  Runs ConservativeSafetyCritic training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--trpo', help='whether to force trpo update', required=False, type=int, default=1)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = Box(json.load(f1))
    config1.seed = in_args.seed
    if 'trpo' not in config1:
        config1.trpo = bool(in_args.trpo)
    if in_args.mode.lower() == 'train':
        csc_object = ConservativeSafetyCritic(config1)
        csc_object.train()
    else:
        config1.use_prior_nets = True
        csc_object = ConservativeSafetyCritic(config1)
        csc_object.test()
