import os
import sys
import json
import argparse
import torch
import numpy as np
from box import Box
from copy import deepcopy
from gentle.common.utils import get_network_object, get_env_object
from gentle.rl.sac import SoftActorCritic
from gentle.common.diagnostics import compute_model_norm, compute_model_grad_norm, get_model_params, compute_update_norm

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OffPolicyActorCritic(SoftActorCritic):

    def __init__(self, config):
        super().__init__(config)
        self.v_network = None
        self.v_optimizer = None
        self.v_target = None
        self.v_params = []
        self.steps = self.config.start_learning

    def process_config(self):
        super().process_config()
        self.config.setdefault('alpha', [0.0, False])
        self.config.setdefault('entropy_type', 'none')  # "reg", "max", or "none"
        self.config.pi_network.setdefault('sampler', 'tanh')
        if self.config.pi_network.sampler != 'tanh':  # "clip"
            self.config.pi_network.setdefault('bound_corr', True)
        self.config.setdefault('compute_H_freq', 10)
        self.config.setdefault('pi_update_skip', 1)
        self.config.algorithm = 'opac'

    def initialize_env(self):
        """  Initialize environment objects  """
        self.env = get_env_object(self.config)
        self.eval_env = get_env_object(self.config)
        if self.config.entropy_type.lower() == 'reg':
            self.config.alpha[0] *= 0.67 * np.prod(self.env.action_space.shape)

    def initialize_networks(self, reset=False):
        """  Initialize network objects  """
        total_steps, last_checkpoint = 0, -1
        if not reset or self.resets < self.config.n_resets_pi:
            self.pi_network = get_network_object(self.config.pi_network).to(device)
        if not reset or self.resets < self.config.n_resets_q:
            self.q1_network = get_network_object(self.config.q_network).to(device)
            self.q_params = list(self.q1_network.parameters())
            self.v_network = get_network_object(self.config.v_network).to(device)
            self.v_target = deepcopy(self.v_network).to(device)
            self.v_params = list(self.v_network.parameters())
        if not reset:
            if self.config.alpha[1]:  # alpha not reset for primacy bias correction
                self.log_alpha = torch.log(torch.ones(1, device=device) * self.config.alpha[0]).requires_grad_(True)
                self.alpha = torch.exp(self.log_alpha.detach())
            else:
                self.alpha = (torch.ones(1, device=device) * self.config.alpha[0]).requires_grad_(False)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.pi_network.load_state_dict(checkpoint.pi)
                self.q1_network.load_state_dict(checkpoint.q1)
                self.v_network.load_state_dict(checkpoint.v)
                if 'v_t' in checkpoint.keys():
                    self.v_target.load_state_dict(checkpoint.v_t)
                if self.config.alpha[1]:
                    self.log_alpha = (torch.ones(1, device=device) * checkpoint.log_a).requires_grad_(True)
                    self.alpha = torch.exp(self.log_alpha.detach())
                total_steps = checkpoint.steps
                last_checkpoint = total_steps // self.config.checkpoint_every
        if self.v_target is not None:
            for p in self.v_target.parameters():
                p.requires_grad = False
        return total_steps, last_checkpoint

    def initialize_optimizers(self, reset=False):
        """  Initializes Adam optimizers for training networks.  """
        if not reset or self.resets < self.config.n_resets_pi:
            self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.lr)
        if not reset or self.resets < self.config.n_resets_q:
            self.q_optimizer = torch.optim.Adam(params=self.q1_network.parameters(), lr=self.config.lr)
            self.v_optimizer = torch.optim.Adam(params=self.v_network.parameters(), lr=self.config.lr)
        if not reset:
            if self.config.alpha[1]:  # alpha not reset
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr_log_alpha)
                self.target_entropy = (-np.prod(self.env.action_space.shape)*self.config.ent_factor).astype(np.float32)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.pi_optimizer.load_state_dict(checkpoint.pi_opt)
                self.q_optimizer.load_state_dict(checkpoint.q_opt)
                self.v_optimizer.load_state_dict(checkpoint.v_opt)
                if self.config.alpha[1]:
                    self.alpha_optimizer.load_state_dict(checkpoint.a_opt)

    def update_networks(self):
        """  Update all networks  """
        # Update value network:
        data = self.buffer.sample(self.config.batch_size)
        old_v_params = get_model_params(self.v_network)
        v_loss, v_info = self.compute_v_loss(data)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        v_info['v_gnorm'] = compute_model_grad_norm(self.v_network).item()
        self.v_optimizer.step()
        v_info['v_unorm'] = compute_update_norm(old_v_params, self.v_network).item()
        # Update Q network:
        old_q_params = get_model_params(self.q1_network)
        q_loss, q_info = self.compute_q_loss(data)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_info['q1_gnorm'] = compute_model_grad_norm(self.q1_network).item()
        self.q_optimizer.step()
        q_info['q1_unorm'] = compute_update_norm(old_q_params, self.q1_network).item()
        # Update policy network:
        pi = self.pi_network(data.obs)
        if self.steps % self.config.pi_update_skip == 0:  # optional; for TD3-style updates
            for p in self.q_params + self.v_params:
                p.requires_grad = False  # fix Q parameters for policy update
            act, log_prob_pi = self.sampler.get_action_and_log_prob(pi)
            old_pi_params = get_model_params(self.pi_network)
            pi_loss, pi_info = self.compute_pi_loss(data, act, log_prob_pi)
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            pi_info['pi_gnorm'] = compute_model_grad_norm(self.pi_network).item()
            self.pi_optimizer.step()
            pi_info['pi_unorm'] = compute_update_norm(old_pi_params, self.pi_network).item()
            for p in self.q_params + self.v_params:
                p.requires_grad = True  # allow Q, V parameters to vary again
        else:
            _, log_prob_pi = self.sampler.get_action_and_log_prob(pi)
            pi_info = {}
        # Update alpha, if required:
        if self.config.alpha[1]:
            alpha_loss = self.compute_alpha_loss(log_prob_pi)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha.detach())
        # Update target networks:
        self.update_v_target()
        # Return info for logging
        with torch.no_grad():
            e_info = {}
            if self.steps % (self.config.epoch_size // self.config.compute_H_freq) == 0:
                e_info = {'gauss entropy': torch.mean(torch.log(pi[1]) + .5 * np.log(2 * np.pi * np.e)).item()}
                _, scale = torch.broadcast_tensors(pi[0], pi[1])
        self.steps += self.config.update_every
        return data, {**v_info, **q_info, **pi_info, **e_info}

    def compute_v_loss(self, data):
        """  Compute value function loss  """
        v_pred = self.v_network(data.obs).squeeze(-1)
        with torch.no_grad():
            pi = self.pi_network(data.obs)
            act, log_prob_pi = self.sampler.get_action_and_log_prob(pi)
            inputs = torch.cat((data.obs, act), dim=-1)
            v_target = self.q1_network(inputs).squeeze(-1)
            if self.config.entropy_type.lower() == 'max':
                v_target -= self.alpha * log_prob_pi
        v_loss = ((v_pred - v_target) ** 2).mean()
        v_info = {'v': torch.mean(v_pred).item(), 'v_pnorm': compute_model_norm(self.v_network).item()}
        return v_loss, v_info

    def compute_q_loss(self, data):
        """ Compute loss for Q update, using either value or Q function  """
        q = self.q1_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        with torch.no_grad():  # just computing target
            v_target = self.v_target(data.next_obs).squeeze(-1)
            not_dones = (torch.ones_like(data.terminated, device=device) - data.terminated).squeeze(-1)
            backup = data.rewards.squeeze(-1) + self.config.gamma * not_dones * v_target
        q_loss = ((q - backup) ** 2).mean()
        q_info = {'q': torch.mean(q).item(), 'q1_pnorm': compute_model_norm(self.q1_network).item()}
        return q_loss, q_info

    def compute_pi_loss(self, data, act, log_prob_pi):
        """  Compute loss for policy network  """
        inputs = torch.cat((data.obs, act), dim=-1)
        with torch.no_grad():
            q_pi = self.q1_network(inputs).squeeze(-1)
            v_pi = self.v_network(data.obs).squeeze(-1)
            advantages = q_pi - v_pi
            if self.config.entropy_type.lower() == 'max':
                advantages -= self.alpha * log_prob_pi
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        pi_loss = torch.mean(-advantages * log_prob_pi)
        if self.config.entropy_type.lower() == 'reg':
            pi = self.pi_network(data.obs)
            _, log_prob_pi_reg = self.sampler.get_action_and_log_prob(pi, reparam=True)
            pi_loss += torch.mean(self.alpha * log_prob_pi_reg)
        pi_info = {'pi_loss': pi_loss.item(), 'pi_pnorm': compute_model_norm(self.pi_network).item()}
        return pi_loss, pi_info

    def update_v_target(self):
        """  Update value target networks via Polyak averaging  """
        with torch.no_grad():
            for p, p_targ in zip(self.v_network.parameters(), self.v_target.parameters()):
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)

    def save_training(self, total_steps, last_checkpoint, data):
        """  Save networks, as required.  Update last_checkpoint.  """
        checkpoint_required = (total_steps in self.config.checkpoint_list or
                               total_steps // self.config.checkpoint_every > last_checkpoint)
        training = Box({})
        if checkpoint_required:
            training = Box({'pi': self.pi_network.state_dict(),
                            'q1': self.q1_network.state_dict(),
                            'v': self.v_network.state_dict(),
                            'steps': total_steps})
            if self.config.alpha[1]:
                training.log_a = self.log_alpha.item()
            if self.config.enable_restart:
                training.v_t = self.v_target.state_dict()
                training.pi_opt = self.pi_optimizer.state_dict()
                training.q_opt = self.q_optimizer.state_dict()
                training.v_opt = self.v_optimizer.state_dict()
                if self.config.alpha[1]:
                    training.a_opt = self.alpha_optimizer.state_dict()
        if total_steps in self.config.checkpoint_list:
            torch.save(training, os.path.join(self.config.model_folder, 'model-' + str(total_steps) + '.pt'))
            if self.config.checkpoint_list_save_batch:
                self.save_batch(data, os.path.join(self.config.model_folder, 'batch-' + str(total_steps) + '.p.tar.gz'))
            if self.config.checkpoint_list_save_buffer:
                self.buffer.save(os.path.join(self.config.model_folder, 'buffer-' + str(total_steps) + '.p.tar.gz'))
        if total_steps // self.config.checkpoint_every > last_checkpoint:
            torch.save(training, os.path.join(self.config.model_folder, 'model-latest.pt'))
            last_checkpoint += 1
            if self.config.enable_restart:  # save buffer
                self.buffer.save(os.path.join(self.config.model_folder, 'buffer-latest.p.tar.gz'))
        return last_checkpoint

    def run_evaluation(self, num_episodes=0, deterministic=False):
        """  Run episodes with deterministic agent, in order to gauge progress  """
        if num_episodes == 0:
            num_episodes = self.config.evaluation_ep
        if num_episodes > 0:
            results = Box({'rewards': [], 'entropies': [], 'lengths': [], 'info': {},
                           'q_pred': [], 'q_true': [], 'v_pred': [], 'v_true': [], 'td_error': []})
        else:
            results = Box({})
        for j in range(num_episodes):
            obs, info = self.reset_env(self.eval_env)
            terminated, truncated = False, False
            ep_rew, ep_len, ep_ent, ep_q, ep_vt, ep_v, ep_info = [], 0, [], [], [], 0, {}
            self.concatenate_dict_of_lists(ep_info, info)
            while not terminated and not truncated:
                action = self.get_action(obs, deterministic=deterministic)
                with torch.no_grad():
                    torch_obs = torch.from_numpy(obs).to(device).float()
                    torch_act = torch.from_numpy(action).to(device).float()
                    pi = self.pi_network(torch_obs)
                    _, log_prob = self.sampler.get_action_and_log_prob(pi)
                    q_pred = self.q1_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    v_targ = self.v_target(torch_obs).item()
                    v_pred = self.v_network(torch_obs).item()
                trunc = ep_len == self.config.max_ep_length - 1
                next_obs, reward, terminated, truncated, info = self.step_env(self.eval_env, action, trunc)
                obs = next_obs
                ep_rew += [reward]
                ep_len += 1
                ep_ent += [torch.mean(-log_prob).item()]
                ep_q += [q_pred]
                ep_vt += [v_targ]
                ep_v += v_pred
                self.concatenate_dict_of_lists(ep_info, info)
            results.rewards.append(sum(ep_rew))
            results.lengths.append(ep_len)
            results.entropies.append(sum(ep_ent)/ep_len)
            for k, v in ep_info.items():
                ep_info[k] = sum(v)
            self.concatenate_dict_of_lists(results.info, ep_info)
            results.q_pred.append(sum(ep_q) / ep_len)
            q_true = self.compute_q(ep_rew, ep_ent)
            results.q_true.append(q_true)
            results.v_pred.append(ep_v / ep_len)
            v_true = self.compute_v(ep_rew, ep_ent)  # v_true = q_true when alpha = 0
            results.v_true.append(v_true)
            td_error = self.compute_td_error(np.array(ep_rew), np.array(ep_q), np.array(ep_vt))
            results.td_error.append(td_error)
        return results

    def compute_q(self, ep_reward, ep_entropy):
        if self.config.entropy_type == 'max':
            full_rewards = np.array(ep_reward) + self.alpha.item() * np.array(ep_entropy)
        else:
            full_rewards = np.array(ep_reward)
        t = len(ep_reward)
        q = [ep_reward[i] + np.sum(self.config.gamma ** np.arange(1, t - i) * full_rewards[i + 1:]) for i in range(t)]
        return sum(q) / t

    def compute_v(self, ep_reward, ep_entropy):
        full_rewards = np.array(ep_reward) + self.alpha.item() * np.array(ep_entropy)
        t = len(ep_reward)
        return sum([np.sum(full_rewards[i:] * self.config.gamma ** np.arange(t-i)) for i in range(t)]) / t

    def compute_td_error(self, ep_reward, ep_q, ep_vt):
        return np.sum((ep_reward[:-1] + self.config.gamma*ep_vt[1:] - ep_q[:-1])**2) / ep_reward.shape[0]


if __name__ == '__main__':
    """  Runs OffPolicyActorCritic training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--prior', help='use prior training', required=False, type=int, default=0)
    in_args = parser.parse_args()
    full_config = os.path.join(os.getcwd(), in_args.config)
    print(full_config)
    sys.stdout.flush()
    with open(full_config, 'r') as f1:
        config1 = Box(json.load(f1))
    config1.seed = in_args.seed
    if in_args.prior > 0 or in_args.mode.lower() == 'test':
        config1.use_prior_nets = True
    opac_object = OffPolicyActorCritic(config1)
    if in_args.mode.lower() == 'train':
        opac_object.train()
    else:
        opac_object.test()
