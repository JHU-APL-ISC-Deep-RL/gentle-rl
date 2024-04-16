import os
import sys
import json
import argparse
import torch
import numpy as np
from box import Box
from copy import deepcopy
from gentle.common.utils import get_network_object
from gentle.common.buffers import RecentCostOffPolicyBuffer
from gentle.common.diagnostics import compute_model_grad_norm, get_model_grads, compute_cos_similarity
from gentle.rl.sac import SoftActorCritic

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConstrainedSoftActorCritic(SoftActorCritic):

    def __init__(self, config):
        super().__init__(config)
        self.qc1_network, self.qc1_target = None, None
        self.qc2_network, self.qc2_target = None, None
        self.beta_param, self.beta = None, None
        self.beta_optimizer = None
        self.target_cost = None

    def process_config(self):
        self.config.store_costs = True
        self.config.setdefault('beta_init', 0.001)  # initial penalty weight (make small to encourage exploration)
        self.config.setdefault('lr_beta', 0.0001)  # learning rate for penalty weight
        self.config.setdefault('cost_limit', 25)  # full-episode cost limit
        self.config.setdefault('two_qc', False)  # whether to use two Qc networks
        self.config.setdefault('beta_recency', -1)  # window to consider for penalty updates
        self.config.setdefault('cost_scale', 1.0)  # rescale cost to ensure it matches reward
        self.config.target_cost = self.config.cost_limit / self.config.max_ep_length * self.config.cost_scale
        self.config.setdefault('log_pi_grads', True)  # whether to log grad components in policy update
        self.config.setdefault('num_actions', 1)  # choose num_actions > 1 to activate safety critic
        super().process_config()

    def initialize_buffer(self):
        """  Initialize replay buffer  """
        self.buffer = RecentCostOffPolicyBuffer(capacity=self.config.buffer_size,
                                                obs_dim=self.env.observation_space.shape[0],
                                                act_dim=self.env.action_space.shape[0],
                                                recent=self.config.beta_recency)
        if self.config.use_prior_nets:
            self.buffer.load(os.path.join(self.config.model_folder, 'buffer-latest.p.tar.gz'))
            self.buffer.episode_lengths = self.buffer.episode_lengths[:-1] + [0]
            self.buffer.episode_costs = self.buffer.episode_costs[:-1] + [0]

    def initialize_networks(self, reset=False):
        """  Initialize network objects  """
        total_steps, last_checkpoint = super().initialize_networks(reset)
        if not reset or self.resets < self.config.n_resets_q:
            self.qc1_network = get_network_object(self.config.q_network).to(device)
            self.qc1_target = deepcopy(self.qc1_network)
            self.q_params += list(self.qc1_network.parameters())
            if self.config.two_qc:
                self.qc2_network = get_network_object(self.config.q_network).to(device)
                self.qc2_target = deepcopy(self.qc2_network)
                self.q_params += list(self.qc2_network.parameters())
        if not reset:
            self.beta_param = (torch.ones(1, device=device) * self.config.beta_init).requires_grad_(True)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.qc1_network.load_state_dict(checkpoint.qc1)
                if 'qc1_t' in checkpoint:
                    self.qc1_target.load_state_dict(checkpoint.qc1_t)
                if self.config.two_qc:
                    self.qc2_network.load_state_dict(checkpoint.qc2)
                    if 'qc2_t' in checkpoint:
                        self.qc2_target.load_state_dict(checkpoint.qc2_t)
                self.beta_param = (torch.ones(1, device=device) * checkpoint.b_param).requires_grad_(True)
            self.beta = torch.nn.functional.relu(self.beta_param).clone().detach()
        if self.q1_target is not None:
            target_params = [self.q1_target, self.q2_target, self.qc1_target]
            if self.config.two_qc:
                target_params += [self.qc2_target]
            for p in torch.nn.ModuleList(target_params).parameters():
                p.requires_grad = False
        return total_steps, last_checkpoint

    def initialize_optimizers(self, reset=False):
        """  Initializes Adam optimizers for training networks  """
        if not reset or self.resets < self.config.n_resets_pi:
            self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.lr)
        if not reset or self.resets < self.config.n_resets_q:
            q_p = [self.q1_network, self.q2_network, self.qc1_network]
            if self.config.two_qc:
                q_p += [self.qc2_network]
            q_params = torch.nn.ModuleList(q_p).parameters()
            self.q_optimizer = torch.optim.Adam(params=q_params, lr=self.config.lr)
        if not reset:
            if self.config.alpha[1]:  # alpha not reset
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr_log_alpha)
                self.target_entropy = (-np.prod(self.env.action_space.shape) * self.config.ent_factor).astype(
                    np.float32)
            self.beta_optimizer = torch.optim.Adam([self.beta_param], lr=self.config.lr_beta)
            self.target_cost = (torch.ones(1, ) * self.config.target_cost).requires_grad_(False)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.pi_optimizer.load_state_dict(checkpoint.pi_opt)
                self.q_optimizer.load_state_dict(checkpoint.q_opt)
                if self.config.alpha[1]:
                    self.alpha_optimizer.load_state_dict(checkpoint.a_opt)
                self.beta_optimizer.load_state_dict(checkpoint.b_opt)

    def get_action(self, obs, random_action=False, deterministic=None):
        """  Get action, optionally integrating a safety critic  """
        with torch.no_grad():
            obs_torch = torch.from_numpy(obs).to(device).float()
            if self.config.num_actions == 1 or random_action:
                pi = self.pi_network(obs_torch)
                return self.sampler.get_action(pi, random_action, deterministic)
            else:  # use safety critic to choose action with minimum Qc
                obs_repeated = obs_torch[None, ...].repeat(self.config.num_actions, 1)
                pi = self.pi_network(obs_repeated)
                act = torch.from_numpy(self.sampler.get_action(pi, False, False)).to(device).float()
                qc_act1 = self.qc1_network(torch.cat((obs_repeated, act), dim=-1)).squeeze(-1)
                qc_act2 = self.qc2_network(torch.cat((obs_repeated, act), dim=-1)).squeeze(-1)
                qc_act = torch.min(torch.max(qc_act1, qc_act2))  # or we could use mean
                return act[torch.argmin(qc_act)].cpu().numpy()

    def update_networks(self):
        """  Update all networks  """
        data = self.buffer.sample(self.config.batch_size)
        pi = self.pi_network(data.obs)
        # Sample action(s) and get log_prob(s) for alpha, pi updates:
        act, log_prob_pi = self.sampler.get_action_and_log_prob(pi)
        # Log entropy:
        with torch.no_grad():
            e_info = {'gauss entropy': torch.mean(torch.log(pi[1]) + .5 * np.log(2 * np.pi * np.e)).item(),
                      'squashed entropy': -log_prob_pi.mean().item()}
        # Update beta (assuming required):
        self.update_beta()
        # Update Q networks:
        q_loss, q_info = self.compute_q_loss(data)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        # Update policy network:
        for p in self.q_params:
            p.requires_grad = False  # fix Q parameters for policy update
        pi_loss, pi_info = self.compute_pi_loss(data, act, log_prob_pi)
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True  # allow Q parameters to vary again
        # Update alpha, if required:
        if self.config.alpha[1]:
            alpha_loss = self.compute_alpha_loss(log_prob_pi)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha.detach())
        # Update target networks:
        self.update_q_targets()
        return data, {**q_info, **pi_info, **e_info}

    def update_beta(self):
        """  Update penalty weight beta  """
        with torch.no_grad():
            current_cost = sum(self.buffer.recent_costs) / len(self.buffer.recent_costs) * self.config.cost_scale
        beta_loss = (self.config.target_cost - current_cost) * self.beta_param
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()
        self.beta = torch.nn.functional.relu(self.beta_param).clone().detach()

    def compute_q_loss(self, data):
        """  Compute loss for Q update  """
        qr1 = self.q1_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        qr2 = self.q2_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        qc1 = self.qc1_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        qc2 = None
        if self.config.two_qc:
            qc2 = self.qc2_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        with torch.no_grad():  # just computing targets
            pi_next = self.pi_network(data.next_obs)
            act_next, log_prob_pi_next = self.sampler.get_action_and_log_prob(pi_next)
            qr1_target = self.q1_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
            qr2_target = self.q2_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
            qr_target = torch.min(qr1_target, qr2_target)
            qc_target = self.qc1_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
            if self.config.two_qc:
                qc2_target = self.qc2_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
                qc_target = torch.max(qc_target, qc2_target)
            not_dones = (torch.ones_like(data.terminated, device=device) - data.terminated).squeeze(-1)
            b_r = data.rewards.squeeze(-1) + self.config.gamma * not_dones * (qr_target - self.alpha * log_prob_pi_next)
            b_c = data.costs.squeeze(-1) * self.config.cost_scale + self.config.gamma * not_dones * qc_target
        qr1_loss = ((qr1 - b_r) ** 2).mean()
        qr2_loss = ((qr2 - b_r) ** 2).mean()
        qc1_loss = ((qc1 - b_c) ** 2).mean()
        if self.config.two_qc:
            qc2_loss = ((qc2 - b_c) ** 2).mean()
        else:
            qc2_loss = torch.zeros(1, device=device)
        q_loss = qr1_loss + qr2_loss + qc1_loss + qc2_loss
        q_info = {'qr1': torch.mean(qr1).item(), 'qr2': torch.mean(qr2).item(),
                  'qc1': torch.mean(qc1).item()}
        if self.config.two_qc:
            q_info.update({'qc2': torch.mean(qc2).item()})
        return q_loss, q_info

    def compute_pi_loss(self, data, act, log_prob_pi):
        """  Compute loss for policy network  """
        inputs = torch.cat((data.obs, act), dim=-1)
        qr1_pi = self.q1_network(inputs).squeeze(-1)
        qr2_pi = self.q2_network(inputs).squeeze(-1)
        qr_pi = torch.min(qr1_pi, qr2_pi)
        qc_pi = self.qc1_network(inputs).squeeze(-1)
        if self.config.two_qc:
            qc2_pi = self.qc2_network(inputs).squeeze(-1)
            qc_pi = torch.max(qc_pi, qc2_pi)
        pi_info = {}
        if self.config.log_pi_grads:
            self.pi_optimizer.zero_grad()
            r_loss = - qr_pi.mean()
            r_loss.backward(retain_graph=True)
            r_grad = get_model_grads(self.pi_network)
            pi_info.update({'r_grad': r_grad.norm().item()})
            self.pi_optimizer.zero_grad()
            c_loss = self.beta * qc_pi.mean()
            c_loss.backward(retain_graph=True)
            c_grad = get_model_grads(self.pi_network)
            pi_info.update({'c_grad': c_grad.norm().item()})
            self.pi_optimizer.zero_grad()
            pi_info.update({'cost_inc_cos': compute_cos_similarity(r_grad, c_grad).item()})
            e_loss = self.alpha * log_prob_pi.mean()
            e_loss.backward(retain_graph=True)
            pi_info.update({'e_grad': compute_model_grad_norm(self.pi_network).item()})
            self.pi_optimizer.zero_grad()  # not strictly necessary; grad is zeroed in update_network
        pi_loss = (self.alpha * log_prob_pi - qr_pi + self.beta * qc_pi).mean()
        pi_info.update({'pi_loss': pi_loss.item()})
        return pi_loss, pi_info

    def update_q_targets(self):
        """  Update target networks via Polyak averaging  """
        super().update_q_targets()
        with torch.no_grad():
            for p, p_targ in zip(self.qc1_network.parameters(), self.qc1_target.parameters()):
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)
            if self.config.two_qc:
                for p, p_targ in zip(self.qc2_network.parameters(), self.qc2_target.parameters()):
                    p_targ.data.mul_(self.config.polyak)
                    p_targ.data.add_((1 - self.config.polyak) * p.data)

    def update_logging(self, loss_info, evaluation_stoch, evaluation_det, steps):
        """  Update TensorBoard logging, reset buffer logging quantities  """
        self.logger.log_mean_value('Learning/beta', [self.beta.item()], steps)
        super().update_logging(loss_info, evaluation_stoch, evaluation_det, steps)

    def save_training(self, total_steps, last_checkpoint, data):
        """  Save networks, as required.  Update last_checkpoint.  """
        checkpoint_required = (total_steps in self.config.checkpoint_list or
                               total_steps // self.config.checkpoint_every > last_checkpoint)
        training = Box({})
        if checkpoint_required:
            training = Box({'pi': self.pi_network.state_dict(),
                            'q1': self.q1_network.state_dict(),
                            'q2': self.q2_network.state_dict(),
                            'qc1': self.qc1_network.state_dict(),
                            'b_param': self.beta_param.item(),
                            'steps': total_steps})
            if self.config.two_qc:
                training.qc2 = self.qc2_network.state_dict()
            if self.config.alpha[1]:
                training.log_a = self.log_alpha.item()
            if self.config.enable_restart:
                training.q1_t = self.q1_target.state_dict()
                training.q2_t = self.q2_target.state_dict()
                training.qc1_t = self.qc1_target.state_dict()
                if self.config.two_qc:
                    training.qc2_t = self.qc2_target.state_dict()
                training.pi_opt = self.pi_optimizer.state_dict()
                training.q_opt = self.q_optimizer.state_dict()
                if self.config.alpha[1]:
                    training.a_opt = self.alpha_optimizer.state_dict()
                training.b_opt = self.beta_optimizer.state_dict()
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
            results = Box({'rewards': [], 'entropies': [], 'max_ent_rew': [], 'lengths': [], 'info': {},
                           'qr1': [], 'qr2': [], 'qr_min': [], 'qr_true': [],
                           'qc1': [], 'qc2': [], 'qc_max': [], 'qc_true': [],
                           'td_error_r1': [], 'td_error_c1': [], 'td_error_r2': []})
            if self.config.two_qc:
                results.update({'td_error_c2': []})
        else:
            results = Box({})
        for j in range(num_episodes):
            obs, info = self.reset_env(self.eval_env)
            terminated, truncated = False, False
            ep_rew, ep_len, ep_ent, ep_q1, ep_q2, ep_q, ep_qt, ep_info = [], 0, [], [], [], 0, [], {}
            ep_cost, ep_qc1, ep_qc2, ep_qc, ep_qct = [], [], [], 0, []
            self.concatenate_dict_of_lists(ep_info, info)
            while not terminated and not truncated:
                action = self.get_action(obs, deterministic=deterministic)
                with torch.no_grad():
                    torch_obs = torch.from_numpy(obs).to(device).float()
                    torch_act = torch.from_numpy(action).to(device).float()
                    pi = self.pi_network(torch_obs)
                    pi = (pi[0][None, ...].repeat(10, 1), pi[1][None, ...].repeat(10, 1))
                    _, log_prob = self.sampler.get_action_and_log_prob(pi)  # only works for stochastic
                    q1_pred = self.q1_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q2_pred = self.q2_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q_pred = min(q1_pred, q2_pred)
                    q1_targ = self.q1_target(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q2_targ = self.q2_target(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q_targ = min(q1_targ, q2_targ)
                    qc1_pred = self.qc1_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    qc1_targ = self.qc1_target(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    if self.config.two_qc:
                        qc2_pred = self.qc2_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                        qc_pred = max(qc1_pred, qc2_pred)
                        qc2_targ = self.qc2_target(torch.cat((torch_obs, torch_act), dim=-1)).item()
                        qc_targ = max(qc1_targ, qc2_targ)
                    else:
                        qc_pred = qc1_pred
                        qc_targ = q1_targ
                trunc = ep_len == self.config.max_ep_length - 1
                next_obs, reward, terminated, truncated, info = self.step_env(self.eval_env, action, trunc)
                obs = next_obs
                ep_rew += [reward]
                ep_cost += [info.get('cost', 0.)]
                ep_len += 1
                ep_ent += [torch.mean(-log_prob).item()]
                ep_q1 += [q1_pred]
                ep_q2 += [q2_pred]
                ep_qt += [q_targ]
                ep_q += q_pred
                ep_qc1 += [qc1_pred]
                if self.config.two_qc:
                    ep_qc2 += [qc2_pred]
                    ep_qc += qc_pred
                ep_qct += [qc_targ]
                self.concatenate_dict_of_lists(ep_info, info)
            results.rewards.append(sum(ep_rew))
            results.lengths.append(ep_len)
            results.entropies.append(sum(ep_ent)/ep_len)
            results.max_ent_rew.append(sum(ep_rew) + self.alpha.item()*sum(ep_ent))
            for k, v in ep_info.items():
                ep_info[k] = sum(v)
            self.concatenate_dict_of_lists(results.info, ep_info)
            results.qr1.append(sum(ep_q1) / ep_len)
            results.qr2.append(sum(ep_q2) / ep_len)
            results.qr_min.append(ep_q / ep_len)
            qr_true = self.compute_q(ep_rew, ep_ent)
            results.qr_true.append(qr_true)
            results.qc1.append(sum(ep_qc1) / ep_len)
            results.qc2.append(sum(ep_qc2) / ep_len)
            results.qc_max.append(ep_qc / ep_len)
            qc_true = self.compute_qc(ep_cost)
            results.qc_true.append(qc_true)
            td_error_r1 = self.compute_td_error(np.array(ep_rew), np.array(ep_q1), np.array(ep_qt))
            results.td_error_r1.append(td_error_r1)
            td_error_r2 = self.compute_td_error(np.array(ep_rew), np.array(ep_q2), np.array(ep_qt))
            results.td_error_r2.append(td_error_r2)
            td_error_c1 = self.compute_td_error(np.array(ep_cost), np.array(ep_qc1), np.array(ep_qct))
            results.td_error_c1.append(td_error_c1)
            if self.config.two_qc:
                td_error_c2 = self.compute_td_error(np.array(ep_cost), np.array(ep_qc2), np.array(ep_qct))
                results.td_error_c2.append(td_error_c2)
        return results

    def compute_qc(self, ep_cost):
        cost_array = np.array(ep_cost)
        t = len(ep_cost)
        return sum([np.sum(cost_array[i:] * self.config.gamma**np.arange(t-i)) for i in range(t)]) / t


if __name__ == '__main__':
    """  Runs ConstrainedSoftActorCritic training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--prior', help='use prior training', required=False, type=int, default=0)
    in_args = parser.parse_args()
    full_config = os.path.join(os.getcwd(), in_args.config)
    print(full_config)
    sys.stdout.flush()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = Box(json.load(f1))
    config1.seed = in_args.seed
    if in_args.prior > 0 or in_args.mode.lower() == 'test':
        config1.use_prior_nets = True
    c_sac_object = ConstrainedSoftActorCritic(config1)
    if in_args.mode.lower() == 'train':
        c_sac_object.train()
    else:
        c_sac_object.test()
