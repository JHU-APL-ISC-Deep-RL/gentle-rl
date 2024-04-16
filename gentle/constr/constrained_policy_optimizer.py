import os
import json
import argparse
import torch
import numpy as np
from mpi4py import MPI
from box import Box
from gentle.common.utils import get_network_object
from gentle.common.samplers import compute_entropy
from gentle.common.mpi_data_utils import average_grads, sync_weights, mpi_statistics_scalar, mpi_avg, \
    print_now, print_zero
from gentle.rl.policy_optimizer import PolicyOptimizer


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class ConstrainedPolicyOptimizer(PolicyOptimizer):

    def __init__(self, config):
        """
        Policy Optimization Agent that either uses policy gradient or surrogate objective in conjunction
        with configurable variance reduction measures.
        """
        super().__init__(config)
        self.c_v_network = None  # optimized concurrently with value function
        self.penalty_param = None
        self.penalty_optimizer = None
        self.penalty = 0

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate  """
        super().process_config()
        # Make sure existing configuration is correct:
        self.config.reward_to_go = True
        self.config.gae = True
        # Process configuration related to constraint:
        self.config.setdefault('cost_limit', 25)                        # this is very low for some environments
        self.config.setdefault('penalty_init', .025)                    # initial value of penalty constant
        self.config.setdefault('penalty_lr', .0001)                     # learning rate for Lagrange multiplier

    def initialize_networks(self):
        """  Initialize network objects  """
        last_checkpoint = -1
        self.pi_network = get_network_object(self.config.pi_network)
        self.v_network = get_network_object(self.config.v_network)
        self.c_v_network = get_network_object(self.config.v_network)  # same config for reward, cost value networks
        self.penalty_param = torch.autograd.Variable(torch.ones(1) * self.config.penalty_init, requires_grad=True)
        if self.config.use_prior_nets:
            if self.mode == 'test' and self.config.test_iteration is not None:
                if self.config.test_iteration < 0:
                    self.pi_network.restore(os.path.join(self.config.model_folder, 'model-latest.pt'))
                    self.v_network.restore(os.path.join(self.config.model_folder, 'value-latest.pt'))
                    self.c_v_network.restore(os.path.join(self.config.model_folder, 'cost-latest.pt'))
                    torch.load(self.penalty_param, os.path.join(self.config.model_folder, 'penalty-latest.pt'))
                else:
                    self.pi_network.restore(os.path.join(self.config.model_folder,
                                                         'model-' + str(self.config.test_iteration) + '.pt'))
                    self.v_network.restore(os.path.join(self.config.model_folder,
                                                        'value-' + str(self.config.test_iteration) + '.pt'))
                    self.c_v_network.restore(os.path.join(self.config.model_folder,
                                                          'cost-' + str(self.config.test_iteration) + '.pt'))
                    self.penalty_param = torch.load(os.path.join(self.config.model_folder, 'penalty-' +
                                                                 str(self.config.test_iteration) + '.pt'))
            else:
                raise NotImplementedError('Currently, use of prior nets only supported in test mode.')
        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.c_v_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.penalty_param)
        return last_checkpoint

    def initialize_optimizers(self):
        """  Initializes Adam optimizer for training network.  Only one worker actually updates parameters.  """
        self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.pi_lr)
        self.v_optimizer = torch.optim.Adam(params=torch.nn.ModuleList([self.v_network, self.c_v_network]).parameters(),
                                            lr=self.config.v_lr)
        if self.id == 0:
            self.penalty_optimizer = torch.optim.Adam(params=[self.penalty_param], lr=self.config.penalty_lr)

    def run_trajectory(self, random_seed=None):
        """  Run trajectories based on current network(s)  """
        episode_buffer, episode_info = np.array([]).reshape(0, 8), {}
        num_frames, episode_cost, entropies = 0, 0, []
        while True:
            policy, value = self.forward_pass()
            action, log_prob = self.sampler.get_action_and_log_prob(policy)
            trunc = num_frames == self.config.max_ep_length - 1
            output_obs, reward, terminated, truncated, info = self.step_env(action, trunc)
            cost = info.get('cost', 0)
            episode_cost += cost
            if self.config.log_info:
                self.concatenate_dict_of_lists(episode_info, info)
            entropies += [compute_entropy(policy)]
            if self.config.render and self.mode == 'test':
                self.env.render()
            num_frames += 1
            episode_buffer = self.update_episode_buffer(episode_buffer, action, reward, policy,
                                                        log_prob, value, terminated + 2 * truncated, cost)
            self.obs = output_obs
            if terminated or truncated:
                self.buffer.log_bootstrap_obs(self.obs)
                if random_seed is not None:
                    self.env.seed(random_seed)  # for testing
                self.obs, _ = self.reset_env()
                self.process_episode(episode_buffer, episode_info)
                if self.mode == 'train':
                    self.buffer.episode_costs = np.append(self.buffer.episode_costs, episode_cost)
                break
        episode_summary = {'episode_reward': np.sum(episode_buffer[:, 2]), 'episode_length': num_frames,
                           'episode_mean_value': np.mean(episode_buffer[:, 5]), 'episode_cost': episode_cost,
                           'episode_entropy': np.mean(entropies), **{k: sum(v) for k, v in episode_info.items()}}
        return episode_summary

    def process_episode(self, episode_buffer, episode_info):
        """  Processes a completed episode, storing required data in buffer  """
        if self.mode == 'train':
            q_values = self.compute_target_values(episode_buffer[:, 2])
            c_q_values = self.compute_target_values(episode_buffer[:, 7])
            self.buffer.update(episode_buffer, q_values=q_values, c_q_values=c_q_values)

    def update_episode_buffer(self, episode_buffer, action, reward, policy, log_prob, value, done, cost=None):
        """  Updates episode buffer for current step  """
        if self.pi_network.config.discrete:
            policy_to_store = np.squeeze(policy.detach().numpy())
        else:
            policy_to_store = np.concatenate((policy[0].detach().numpy(), policy[1].detach().numpy()))
        experience = np.reshape(np.array([self.obs, action, reward, policy_to_store, log_prob, value, done, cost],
                                         dtype=object), (1, 8))
        return np.concatenate((episode_buffer, experience))

    def update_penalty(self):
        """  Updates Lagrange multiplier based on current batch of data  """
        mean_cost, _ = mpi_statistics_scalar(MPI.COMM_WORLD, self.buffer.episode_costs)
        if self.id == 0:
            self.penalty_optimizer.zero_grad()
            penalty_loss = (self.config.cost_limit - mean_cost) * self.penalty_param
            penalty_loss.backward()
            self.penalty_optimizer.step()
        sync_weights(MPI.COMM_WORLD, self.penalty_param)
        self.penalty = torch.nn.functional.relu(self.penalty_param).item()

    def update_network(self):
        """  Updates the networks based on processing from all workers  """
        self.pi_network.eval()
        self.v_network.eval()
        self.c_v_network.eval()
        # Update Lagrange multiplier:
        self.update_penalty()
        # Update value network:
        observations = torch.from_numpy(np.vstack(self.buffer.observations)).float()
        values = torch.from_numpy(self.buffer.values.astype(float)).float()
        pi_losses, v_losses, entropies = [], [], []
        # Update reward, cost value functions:
        for i in range(self.config.train_v_iter):
            self.v_optimizer.zero_grad()
            target_values = torch.from_numpy(self.buffer.q_values.astype(float)).float()
            target_cost_values = torch.from_numpy(self.buffer.c_q_values.astype(float)).float()
            r_v_loss = self.compute_value_loss(observations, target_values, values)
            c_v_loss = self.compute_cost_value_loss(observations, target_cost_values)
            v_loss = r_v_loss + c_v_loss
            v_losses.append(v_loss.item())
            v_loss.backward()
            average_grads(MPI.COMM_WORLD, self.v_network.parameters())
            average_grads(MPI.COMM_WORLD, self.c_v_network.parameters())
            if self.id == 0:
                self.v_optimizer.step()
            sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
            sync_weights(MPI.COMM_WORLD, self.c_v_network.parameters())
        # Update advantage estimates, standardizing across workers:
        self.update_values(observations)
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
            return {'pi_losses': pi_losses, 'v_losses': v_losses,
                    'entropies': entropies, 'penalty': [self.penalty]}
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
            return {'pi_losses': [final_loss], 'v_losses': v_losses,
                    'entropies': [entropy.item()], 'penalty': [self.penalty]}

    def compute_cost_value_loss(self, observations, target_cost_values):
        """  Compute cost value function loss  """
        new_cost_values = self.c_v_network(observations).view(-1)
        cost_value_loss = torch.mean(torch.pow(new_cost_values - target_cost_values, 2), dim=0)
        return cost_value_loss

    def update_values(self, observations):
        """  Update values in buffer, including bootstraps  """
        with torch.no_grad():
            self.buffer.values = self.v_network(observations).view(-1).numpy()  # reward values
            self.buffer.cost_values = self.c_v_network(observations).view(-1).numpy()  # cost values
            # compute bootstraps:
            torch_bootstrap_obs = torch.from_numpy(self.buffer.bootstrap_obs).float()
            bootstrap_values = self.v_network(torch_bootstrap_obs).squeeze(-1).numpy()
            bootstrap_cost_values = self.c_v_network(torch_bootstrap_obs).squeeze(-1).numpy()
            bootstrap_values -= self.penalty * bootstrap_cost_values
            terminals = np.nonzero(self.buffer.dones.astype(int))[0]
            self.buffer.bootstraps = bootstrap_values * (terminals > 1)

    def estimate_advantages(self):
        """  Estimate advantages for a sequence of observations and rewards  """
        rewards = self.buffer.rewards - self.penalty * self.buffer.costs
        values = self.buffer.values - self.penalty * self.buffer.cost_values
        dones = self.buffer.dones
        self.buffer.advantages = self.estimate_generalized_advantage(rewards, values, dones)
        mean_adv, std_adv = mpi_statistics_scalar(MPI.COMM_WORLD, self.buffer.advantages)
        self.buffer.advantages = (self.buffer.advantages - mean_adv) / std_adv
        return mean_adv, std_adv

    def save_networks(self, total_steps, last_checkpoint):
        """  Save networks, as required.  Update last_checkpoint.  """
        self.pi_network.save(os.path.join(self.config.model_folder, 'model-latest.pt'))
        self.v_network.save(os.path.join(self.config.model_folder, 'value-latest.pt'))
        self.c_v_network.save(os.path.join(self.config.model_folder, 'cost-latest.pt'))
        torch.save(self.penalty_param, os.path.join(self.config.model_folder, 'penalty-latest.pt'))
        if total_steps // self.config.checkpoint_every > last_checkpoint:  # Periodically keep checkpoint
            last_checkpoint += 1
            suffix = str(int(last_checkpoint * self.config.checkpoint_every))
            self.pi_network.save(os.path.join(self.config.model_folder, 'model-' + suffix + '.pt'))
            self.v_network.save(os.path.join(self.config.model_folder, 'value-' + suffix + '.pt'))
            self.c_v_network.save(os.path.join(self.config.model_folder, 'cost-' + suffix + '.pt'))
            torch.save(self.penalty_param, os.path.join(self.config.model_folder, 'penalty-' + suffix + '.pt'))
        return last_checkpoint

    def compute_metrics(self, episode_data):
        """  Computes metrics to be evaluated as learning progresses  """
        mean_total = (sum(episode_data['episode_reward']) - self.penalty * sum(episode_data['episode_cost']))\
            / len(episode_data)
        return {'mean': mean_total}


if __name__ == '__main__':
    """  Runs ConstrainedPolicyOptimizer training or testing for a given input configuration file  """
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
        cpg_object = ConstrainedPolicyOptimizer(config1)
        cpg_object.train()
    else:
        config1.use_prior_nets = True
        cpg_object = ConstrainedPolicyOptimizer(config1)
        cpg_object.test()
