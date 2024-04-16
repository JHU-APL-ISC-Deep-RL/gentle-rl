import os
import json
import argparse
import torch
import numpy as np
from mpi4py import MPI
from box import Box
from gentle.common.mpi_data_utils import average_grads, sync_weights, mpi_avg, print_now
from gentle.constr.constrained_policy_optimizer import ConstrainedPolicyOptimizer


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class FOCOPS(ConstrainedPolicyOptimizer):

    def process_config(self):
        self.config.setdefault('nu_opt', 'adam')  # or sgd
        self.config.setdefault('lam', 1.5)
        self.config.clip = -1
        super().process_config()

    def initialize_optimizers(self):
        """  Initializes Adam optimizer for training network.  Only one worker actually updates parameters.  """
        super().initialize_optimizers()
        if self.id == 0:
            if not self.config.nu_opt.lower() == 'adam':
                self.penalty_optimizer = torch.optim.SGD(params=[self.penalty_param], lr=self.config.penalty_lr)

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
        self.update_values(observations)
        self.update_cost_values(observations)
        # Update advantage estimates, standardizing across workers:
        self.estimate_advantages()
        advantages = torch.from_numpy(self.buffer.advantages.astype(float)).float()
        # Update policy network:
        actions = self.buffer.actions
        old_log_probs = torch.from_numpy(self.buffer.log_probs.astype(float)).float()
        for i in range(self.config.train_pi_iter):
            self.pi_optimizer.zero_grad()
            partial_pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages, old_log_probs)
            pi_loss = kld + 1 / self.config.lam * torch.mean(partial_pi_loss, dim=0)
            pi_losses.append(pi_loss.item())
            entropies.append(entropy.item())
            mean_kld = mpi_avg(MPI.COMM_WORLD, kld.item())
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

    def compute_kld(self, policy_predictions, old_log_probs, new_log_probs):
        """ Compute KL divergence for early stopping  """
        new_dist = self.sampler.get_distribution(policy_predictions)
        old_dist = self.sampler.restore_distribution(self.buffer.policies)
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()


if __name__ == '__main__':
    """  Runs FOCOPS training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--trpo', help='whether to force trpo update', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = Box(json.load(f1))
    config1.seed = in_args.seed
    if in_args.mode.lower() == 'train':
        focops_object = FOCOPS(config1)
        focops_object.train()
    else:
        config1.use_prior_nets = True
        focops_object = FOCOPS(config1)
        focops_object.test()
