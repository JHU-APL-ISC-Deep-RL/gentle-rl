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
from gentle.rl.opac2 import OffPolicyActorCritic

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConstrainedOffPolicyActorCritic(OffPolicyActorCritic):

    def __init__(self, config):
        self.qc1_network = None
        self.vc_network, self.vc_target = None, None
        self.beta_param, self.beta = None, None
        self.beta_optimizer = None
        self.target_cost = None
        super().__init__(config)

    def process_config(self):
        self.config.store_costs = True
        self.config.setdefault('beta_init', 0.001)  # initial penalty weight (make small to encourage exploration)
        self.config.setdefault('lr_beta', 0.00001)  # learning rate for penalty weight
        self.config.setdefault('cost_limit', 25)  # full-episode cost limit
        self.config.setdefault('cost_scale', 1.0)  # rescale cost to ensure it matches reward
        self.config.target_cost = self.config.cost_limit / self.config.max_ep_length * self.config.cost_scale
        self.config.setdefault('beta_recency', 10000)  # window to consider for penalty updates
        self.config.setdefault('log_pi_grads', True)  # whether to log grad components in policy update
        self.config.setdefault('num_actions', 1)  # choose num_actions > 1 to activate safety critic
        self.config.setdefault('num_cql', 100)  # number of actions sampled for conservative Q learning
        self.config.setdefault('conservative', False)  # whether to learn Qc conservatively
        self.config.setdefault('cql_alpha', 5)  # alpha weight for CQL
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
            self.buffer.episode_rewards = self.buffer.episode_rewards[:-1] + [0]
            self.buffer.episode_costs = self.buffer.episode_costs[:-1] + [0]

    def initialize_networks(self, reset=False):
        """  Initialize network objects  """
        total_steps, last_checkpoint = super().initialize_networks(reset)
        if not reset or self.resets < self.config.n_resets_q:
            self.qc1_network = get_network_object(self.config.q_network).to(device)
            self.q_params += list(self.qc1_network.parameters())
            self.vc_network = get_network_object(self.config.v_network).to(device)
            self.vc_target = deepcopy(self.vc_network).to(device)
            self.v_params += list(self.vc_network.parameters())
        if not reset:
            self.beta_param = (torch.ones(1, device=device) * self.config.beta_init).requires_grad_(True)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.qc1_network.load_state_dict(checkpoint.qc1)
                self.vc_network.load_state_dict(checkpoint.vc)
                if 'vc_t' in checkpoint:
                    self.vc_target.load_state_dict(checkpoint.vc_t)
                self.beta_param = (torch.ones(1, device=device) * checkpoint.b_param).requires_grad_(True)
            self.beta = torch.nn.functional.relu(self.beta_param).clone().detach()
        if self.v_target is not None:
            for p in torch.nn.ModuleList([self.v_target, self.vc_target]).parameters():
                p.requires_grad = False
        return total_steps, last_checkpoint

    def initialize_optimizers(self, reset=False):
        """  Initializes Adam optimizers for training networks  """
        if not reset or self.resets < self.config.n_resets_pi:
            self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.lr)
        if not reset or self.resets < self.config.n_resets_q:
            q_params = torch.nn.ModuleList([self.q1_network, self.qc1_network]).parameters()
            self.q_optimizer = torch.optim.Adam(params=q_params, lr=self.config.lr)
            v_params = torch.nn.ModuleList([self.v_network, self.vc_network]).parameters()
            self.v_optimizer = torch.optim.Adam(params=v_params, lr=self.config.lr)
        if not reset:
            if self.config.alpha[1]:  # alpha not reset
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr_log_alpha)
                self.target_entropy = (-np.prod(self.env.action_space.shape)*self.config.ent_factor).astype(np.float32)
            self.beta_optimizer = torch.optim.Adam([self.beta_param], lr=self.config.lr_beta)
            self.target_cost = (torch.ones(1, ) * self.config.target_cost).requires_grad_(False)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.pi_optimizer.load_state_dict(checkpoint.pi_opt)
                self.q_optimizer.load_state_dict(checkpoint.q_opt)
                self.v_optimizer.load_state_dict(checkpoint.v_opt)
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
                qc_act = self.qc1_network(torch.cat((obs_repeated, act), dim=-1)).squeeze(-1)
                return act[torch.argmin(qc_act)].cpu().numpy()

    def update_networks(self):
        """  Update all networks  """
        # Update beta:
        data = self.buffer.sample(self.config.batch_size)
        pi = self.pi_network(data.obs)
        self.update_beta()
        # Update value network:
        data = self.buffer.sample(self.config.batch_size)
        v_loss, v_info = self.compute_v_loss(data)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        # Update Q network:
        q_loss, q_info = self.compute_q_loss(data)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        # Update policy network:
        if self.steps % self.config.pi_update_skip == 0:  # optional; for TD3-style updates
            pi = self.pi_network(data.obs)
            act, log_prob_pi = self.sampler.get_action_and_log_prob(pi)
            for p in self.q_params + self.v_params:
                p.requires_grad = False  # fix Q parameters for policy update
            pi_loss, pi_info = self.compute_pi_loss(data, act, log_prob_pi)
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            self.pi_optimizer.step()
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
        self.steps += self.config.update_every
        return data, {**v_info, **q_info, **pi_info, **e_info}

    def update_beta(self):
        """  Update penalty weight beta  """
        with torch.no_grad():
            current_cost = sum(self.buffer.recent_costs) / len(self.buffer.recent_costs) * self.config.cost_scale
        beta_loss = (self.config.target_cost - current_cost) * self.beta_param
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()
        self.beta = torch.nn.functional.relu(self.beta_param).clone().detach()

    def compute_v_loss(self, data):
        """  Compute value function loss  """
        vr_pred = self.v_network(data.obs).squeeze(-1)
        vc_pred = self.vc_network(data.obs).squeeze(-1)
        with torch.no_grad():
            pi = self.pi_network(data.obs)
            act, log_prob_pi = self.sampler.get_action_and_log_prob(pi)
            inputs = torch.cat((data.obs, act), dim=-1)
            vr_target = self.q1_network(inputs).squeeze(-1)
            if self.config.entropy_type.lower() == 'max':
                vr_target -= self.alpha * log_prob_pi
            vc_target = self.qc1_network(inputs).squeeze(-1)
        vr_loss = ((vr_pred - vr_target) ** 2).mean()
        vc_loss = ((vc_pred - vc_target) ** 2).mean()
        v_loss = vr_loss + vc_loss
        v_info = {'vr': torch.mean(vr_pred).item(), 'vc': torch.mean(vc_pred).item()}
        return v_loss, v_info

    def compute_q_loss(self, data):
        """ Compute loss for Q update, using either value or Q function  """
        qr_loss, q_info = super().compute_q_loss(data)
        qc = self.qc1_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        with torch.no_grad():  # just computing target
            vc_target = self.vc_target(data.next_obs).squeeze(-1)
            not_dones = (torch.ones_like(data.terminated, device=device) - data.terminated).squeeze(-1)
            backup = data.costs.squeeze(-1) * self.config.cost_scale + self.config.gamma * not_dones * vc_target
        qc_loss = ((qc - backup) ** 2).mean()
        if self.config.conservative:  # CQL contribution
            qcl_loss = self.compute_cql_loss(data, qc)
            qc_loss += qcl_loss
        q_info.update({'qc': torch.mean(qc).item()})
        return qr_loss + qc_loss, q_info

    def compute_cql_loss(self, data, q_data):
        """  Returns CQL contribution to Qc loss; maximizes Qc from new policy, minimizes from data  """
        repeated_current_obs = self.repeat_obs(data.obs, self.config.num_cql)  # [B*R, O]
        # current obs, current actions:
        pi_current = self.pi_network(repeated_current_obs)  # [B*R, A], [B*R, A]
        act_current, _ = self.sampler.get_action_and_log_prob(pi_current)  # [B*R, A], [B*R]
        qc_current = self.qc1_network(torch.cat((repeated_current_obs, act_current),
                                                dim=-1)).view(self.config.batch_size, -1)  # [B, R]
        cql_loss = self.config.cql_alpha * (q_data - qc_current.mean(dim=-1)).mean()
        return cql_loss

    def compute_pi_loss(self, data, act, log_prob_pi):
        """  Compute loss for policy network  """
        inputs = torch.cat((data.obs, act), dim=-1)
        with torch.no_grad():
            qr_pi = self.q1_network(inputs).squeeze(-1)
            vr_pi = self.v_network(data.obs).squeeze(-1)
            qc_pi = self.qc1_network(inputs).squeeze(-1)
            vc_pi = self.vc_network(data.obs).squeeze(-1)
            advantages = qr_pi - vr_pi - self.beta * (qc_pi - vc_pi)
            if self.config.entropy_type.lower() == 'max':
                advantages -= self.alpha * log_prob_pi
            mean_advantages = torch.mean(advantages)
            std_advantages = torch.std(advantages)
            advantages = (advantages - mean_advantages) / std_advantages
        pi_info = {}
        if self.config.log_pi_grads:
            e_advantages = 0
            with torch.no_grad():
                r_advantages = qr_pi - vr_pi
                c_advantages = -self.beta * (qc_pi - vc_pi)
                r_advantages = (r_advantages - torch.mean(r_advantages)) / std_advantages
                c_advantages = (c_advantages - torch.mean(c_advantages)) / std_advantages
                if self.config.entropy_type.lower() == 'max':
                    e_advantages = -self.alpha * log_prob_pi
                    e_advantages = (e_advantages - torch.mean(e_advantages)) / std_advantages
            self.pi_optimizer.zero_grad()
            r_loss = torch.mean(-r_advantages * log_prob_pi)
            r_loss.backward(retain_graph=True)
            r_grad = get_model_grads(self.pi_network)
            pi_info.update({'r_grad': r_grad.norm().item()})
            self.pi_optimizer.zero_grad()
            c_loss = torch.mean(-c_advantages * log_prob_pi)
            c_loss.backward(retain_graph=True)
            c_grad = get_model_grads(self.pi_network)
            pi_info.update({'c_grad': c_grad.norm().item()})
            pi_info.update({'cost_inc_cos': compute_cos_similarity(r_grad, c_grad).item()})
            if self.config.entropy_type.lower() == 'max':
                self.pi_optimizer.zero_grad()
                e_loss = torch.mean(-e_advantages * log_prob_pi)
                e_loss.backward(retain_graph=True)
                pi_info.update({'e_grad': compute_model_grad_norm(self.pi_network).item()})
            self.pi_optimizer.zero_grad()
        pi_loss = torch.mean(-advantages * log_prob_pi)
        if self.config.entropy_type.lower() == 'reg':
            pi = self.pi_network(data.obs)
            _, log_prob_pi_reg = self.sampler.get_action_and_log_prob(pi, reparam=True)
            if self.config.log_pi_grads:
                e_loss = torch.mean(self.alpha * log_prob_pi_reg)
                e_loss.backward(retain_graph=True)
                pi_info.update({'e_grad': compute_model_grad_norm(self.pi_network).item()})
                self.pi_optimizer.zero_grad()
            pi_loss += torch.mean(self.alpha * log_prob_pi_reg)
        pi_info.update({'pi_loss': pi_loss.item()})
        return pi_loss, pi_info

    def update_v_target(self):
        """  Update value target networks via Polyak averaging  """
        super().update_v_target()
        with torch.no_grad():
            for p, p_targ in zip(self.vc_network.parameters(), self.vc_target.parameters()):
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
                            'qc1': self.qc1_network.state_dict(),
                            'v': self.v_network.state_dict(),
                            'vc': self.vc_network.state_dict(),
                            'b_param': self.beta_param.item(),
                            'steps': total_steps})
            if self.config.alpha[1]:
                training.log_a = self.log_alpha.item()
            if self.config.enable_restart:
                training.v_t = self.v_target.state_dict()
                training.vc_t = self.vc_target.state_dict()
                training.pi_opt = self.pi_optimizer.state_dict()
                training.q_opt = self.q_optimizer.state_dict()
                training.v_opt = self.v_optimizer.state_dict()
                training.b_opt = self.beta_optimizer.state_dict()
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
            results = Box({'rewards': [], 'entropies': [], 'max_ent_rew': [], 'lengths': [], 'info': {},
                           'q_pred': [], 'q_true': [], 'v_pred': [], 'v_true': [],
                           'qc_pred': [], 'qc_true': [], 'vc_pred': [], 'td_r': [], 'td_c': []})
        else:
            results = Box({})
        for j in range(num_episodes):
            obs, info = self.reset_env(self.eval_env)
            terminated, truncated = False, False
            ep_rew, ep_len, ep_ent, ep_q, ep_v, ep_info = [], 0, [], [], 0, {}
            ep_cost, ep_qc, ep_vc, ep_vt, ep_vtc = [], [], 0, [], []
            self.concatenate_dict_of_lists(ep_info, info)
            while not terminated and not truncated:
                action = self.get_action(obs, deterministic=deterministic)
                with torch.no_grad():
                    torch_obs = torch.from_numpy(obs).to(device).float()
                    torch_act = torch.from_numpy(action).to(device).float()
                    pi = self.pi_network(torch_obs)
                    pi = (pi[0][None, ...].repeat(10, 1), pi[1][None, ...].repeat(10, 1))
                    _, log_prob = self.sampler.get_action_and_log_prob(pi)
                    q_pred = self.q1_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    v_pred = self.v_network(torch_obs).item()
                    qc_pred = self.qc1_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    vc_pred = self.vc_network(torch_obs).item()
                    v_targ = self.v_target(torch_obs).item()
                    vc_targ = self.vc_target(torch_obs).item()
                trunc = ep_len == self.config.max_ep_length - 1
                next_obs, reward, terminated, truncated, info = self.step_env(self.eval_env, action, trunc)
                obs = next_obs
                ep_rew += [reward]
                ep_cost += [info.get('cost', 0.)]
                ep_len += 1
                ep_ent += [torch.mean(-log_prob).item()]
                ep_q += [q_pred]
                ep_v += v_pred
                ep_qc += [qc_pred]
                ep_vc += vc_pred
                ep_vt += [v_targ]
                ep_vtc += [vc_targ]
                self.concatenate_dict_of_lists(ep_info, info)
            results.rewards.append(sum(ep_rew))
            results.lengths.append(ep_len)
            results.entropies.append(sum(ep_ent)/ep_len)
            results.max_ent_rew.append(sum(ep_rew) + self.alpha.item()*sum(ep_ent))
            for k, v in ep_info.items():
                ep_info[k] = sum(v)
            self.concatenate_dict_of_lists(results.info, ep_info)
            results.q_pred.append(sum(ep_q) / ep_len)
            q_true = self.compute_q(ep_rew, ep_ent)
            results.q_true.append(q_true)
            results.v_pred.append(ep_v / ep_len)
            v_true = self.compute_v(ep_rew, ep_ent)  # v_true = q_true when alpha = 0
            results.v_true.append(v_true)
            results.qc_pred.append(sum(ep_qc) / ep_len)
            qc_true = self.compute_qc(ep_cost)
            results.qc_true.append(qc_true)
            results.vc_pred.append(ep_vc / ep_len)  # vc_true is the same as qc_true
            td_r = self.compute_td_error(np.array(ep_rew), np.array(ep_q), np.array(ep_vt))
            results.td_r.append(td_r)
            td_c = self.compute_td_error(np.array(ep_cost), np.array(ep_qc), np.array(ep_vtc))
            results.td_c.append(td_c)
        return results

    def compute_qc(self, ep_cost):
        cost_array = np.array(ep_cost)
        t = len(ep_cost)
        return sum([np.sum(cost_array[i:] * self.config.gamma**np.arange(t - i)) for i in range(t)]) / t

    @staticmethod
    def repeat_obs(obs, num_repeat):
        num_obs = obs.shape[0]  # B
        repeated_obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(num_obs * num_repeat, -1)  # [B*R, O]
        return repeated_obs


if __name__ == '__main__':
    """  Runs ConstrainedOffPolicyActorCritic training or testing for a given input configuration file  """
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
    c_opac2_object = ConstrainedOffPolicyActorCritic(config1)
    if in_args.mode.lower() == 'train':
        c_opac2_object.train()
    else:
        c_opac2_object.test()
