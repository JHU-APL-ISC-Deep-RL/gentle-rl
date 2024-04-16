import torch
import numpy as np
from abc import ABCMeta, abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_entropy(policy, lb=-1, ub=1, inc=0.01):
    mean = policy[0].detach().numpy()
    std = policy[1].detach().numpy()

    def prob(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** .5

    entropies = []
    for i in range(len(mean)):
        bins = np.arange(mean[i] - 4 * std[i], mean[i] + 4 * std[i], inc)
        probs = prob(bins, mean[i], std[i])
        mp = []
        for j in range(len(bins)):
            if lb < bins[j] < ub:
                mp.append(probs[j])
        mp = np.array(mp)
        entropies.append(-np.sum(mp * inc * np.log(mp)))
    return np.sum(entropies)


class BaseSampler(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, deterministic):
        self.epsilon = 1.e-6
        self.config = config
        self.deterministic = deterministic

    def get_raw_action(self, action):
        """  Convert action seen by the environment to raw action sampled from network  """
        return action

    @abstractmethod
    def get_action_and_log_prob(self, pi):
        """  Sample action based on neural network output  """
        raise NotImplementedError('Sampler must have get_action_and_log_prob method.')

    def compute_kld(self, old_pi, new_pi):
        """  Return average KL divergence between old and new policies  """
        pass
        # raise NotImplementedError('Sampler must have compute_kld method.')

    @staticmethod
    def get_distribution(pi):
        """  Return Torch distribution object as described by policy  """
        pass

    @staticmethod
    def compute_entropy(pi):
        """  Return average entropy of distributions described by policy  """
        pass

    @staticmethod
    def np_entropy(pi):
        """  Returns entropies of each row of a numpy array of policies  """
        pass


class CategoricalSampler(BaseSampler):

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi = pi.detach().numpy()
            if len(pi.shape) > 1:
                pi = np.squeeze(pi)
            action = int(np.random.choice(np.arange(pi.shape[-1]), p=pi))
            log_prob = np.log(pi[action])
        else:
            log_prob = None  # Not needed for testing
            action = int(np.argmax(pi.detach().numpy()))
        return action, log_prob

    def compute_kld(self, old_pi, new_pi):
        new_pi = new_pi.detach().numpy()
        all_terms = new_pi * (np.log(new_pi) - np.log(old_pi))
        return np.mean(np.sum(all_terms, axis=1))

    @staticmethod
    def compute_entropy(pi):
        return -torch.mean(pi * torch.log(pi))

    @staticmethod
    def np_entropy(pi):
        raise NotImplementedError('Not yet implemented!')


class GaussianSampler(BaseSampler):

    def __init__(self, config, deterministic):
        super().__init__(config, deterministic)
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)
        self.config.setdefault('bound_corr', False)

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi_distribution = self.get_distribution(pi)
            sample = pi_distribution.sample()
            action = sample.detach().numpy()
            if self.config['bound_corr']:
                below = pi_distribution.cdf(sample.clamp(-1.0, 1.0))
                below = below.clamp(self.epsilon, 1.0-self.epsilon).log().detach().numpy()*(action <= -1).astype(float)
                above = (torch.ones(sample.size()) - pi_distribution.cdf(sample.clamp(-1.0, 1.0)))
                above = above.clamp(self.epsilon, 1.0-self.epsilon).log().detach().numpy() * (action >= 1).astype(float)
                in_bounds = pi_distribution.log_prob(sample).detach().numpy() * \
                    ((action > -1).astype(float)) * ((action < 1).astype(float))
                log_prob = np.sum(in_bounds + below + above)
            else:
                log_prob = torch.sum(pi_distribution.log_prob(sample), dim=-1).detach().numpy()
        else:
            log_prob = None  # Not needed for testing
            action = pi[0].detach().numpy()
        return action * self.config['act_scale'] + self.config['act_offset'], log_prob

    def get_raw_action(self, action):
        return (action - self.config['act_offset']) / self.config['act_scale']

    def compute_kld(self, old_pi, new_pi):
        mu_old, sigma_old = np.split(old_pi, 2, axis=1)
        mu_new, sigma_new = new_pi[0].detach().numpy(), new_pi[1].detach().numpy()
        if not self.config['log_std_net']:
            sigma_new = np.repeat(np.expand_dims(sigma_new, 0), sigma_old.shape[0], axis=0)
        var_old, var_new = sigma_old ** 2, sigma_new ** 2
        all_kld = np.log(sigma_new / sigma_old) + 0.5 * (((mu_new - mu_old) ** 2 + var_old) / (var_new + 1.e-8) - 1)
        return np.mean(np.sum(all_kld, axis=1))

    def get_action(self, pi, random=False, deterministic=None):
        if random:
            return (2*np.random.rand(*pi[0].cpu().numpy().shape)-1)*self.config['act_scale'] + self.config['act_offset']
        if deterministic is None:
            deterministic = self.deterministic
        if not deterministic:
            pi_distribution = self.get_distribution(pi)
            action = pi_distribution.sample()  # don't need reparameterization trick
            action = action.cpu().numpy() * self.config['act_scale'] + self.config['act_offset']
        else:
            action = pi[0].cpu().numpy() * self.config['act_scale'] + self.config['act_offset']
        return action

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Normal(pi[0], pi[1])

    @staticmethod
    def restore_distribution(pi):
        pi = np.split(pi, 2, axis=1)
        return torch.distributions.Normal(torch.from_numpy(pi[0]), torch.from_numpy(pi[1]))

    @staticmethod
    def compute_entropy(pi):
        return torch.mean(torch.log(pi[1]) + .5 * np.log(2 * np.pi * np.e))

    @staticmethod
    def np_entropy(pi):
        return np.mean(np.log(np.split(pi, 2, axis=1)[1]), axis=1) + .5 * np.log(2 * np.pi * np.e)


class SoftActorCriticSampler(BaseSampler):

    def __init__(self, config, deterministic):
        super().__init__(config, deterministic)
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)
        self.config.setdefault('min_mu', None)  # CQL version of rlkit clamps to [-9, 9]
        self.config.setdefault('max_mu', None)

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Normal(pi[0], pi[1])

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi_distribution = self.get_distribution(pi)
            action = pi_distribution.rsample()
            if self.config.max_mu and self.config.min_mu:
                action = torch.clamp(action, self.config.min_mu, self.config.max_mu)
            log_prob = pi_distribution.log_prob(action).sum(dim=-1)
            squash = (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))).sum(dim=-1)
            log_prob -= squash
            action = torch.tanh(action) * self.config['act_scale'] + self.config['act_offset']
        else:
            action = torch.tanh(pi[0]).numpy() * self.config['act_scale'] + self.config['act_offset']
            log_prob = -torch.ones((1,), device=device)  # Not needed for testing
        return action, log_prob

    def get_action(self, pi, random=False, deterministic=None):
        if random:
            return (2*np.random.rand(*pi[0].cpu().numpy().shape)-1)*self.config['act_scale'] + self.config['act_offset']
        if deterministic is None:
            deterministic = self.deterministic
        if not deterministic:
            pi_distribution = self.get_distribution(pi)
            action = pi_distribution.sample()  # don't need reparameterization trick
            action = torch.tanh(action).cpu().numpy() * self.config['act_scale'] + self.config['act_offset']
        else:
            action = torch.tanh(pi[0]).cpu().numpy() * self.config['act_scale'] + self.config['act_offset']
        return action


class TD3Sampler(BaseSampler):

    def __init__(self, config, action_space):
        super().__init__(config, None)
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)
        self.action_low = torch.from_numpy(action_space.low).float()
        self.action_high = torch.from_numpy(action_space.high).float()
        self.max_action = torch.cat((self.action_high, self.action_low)).abs().max()

    def get_action(self, pi, noise_scale, random=False, noise_clip=None, clamp_action=True):
        if random:
            return (2 * torch.rand(*pi.shape) - 1) * self.config['act_scale'] + self.config[
                'act_offset']
        action = pi * self.config['act_scale'] + self.config['act_offset']

        noise = noise_scale * self.max_action * torch.randn_like(action)  # Noise scale is relative to max action
        if noise_clip:
            noise = torch.clamp(noise, -noise_clip * self.max_action.item(), noise_clip * self.max_action.item())
        action += noise

        if clamp_action:
            action = torch.clamp(action, self.action_low[0].item(), self.action_high[0].item())
        return action


class OffPolicyActorCriticClipSampler(BaseSampler):

    def __init__(self, config, deterministic):
        super().__init__(config, deterministic)
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Normal(pi[0], pi[1])

    def get_action(self, pi, random=False, deterministic=None):
        if random:
            return (2*np.random.rand(*pi[0].numpy().shape) - 1) * self.config['act_scale'] + self.config['act_offset']
        if deterministic is None:
            deterministic = self.deterministic
        if not deterministic:
            pi_distribution = self.get_distribution(pi)
            action = pi_distribution.sample()
            action = action.clamp(-1.0, 1.0).numpy() * self.config['act_scale'] + self.config['act_offset']
        else:
            action = pi[0].clamp(-1.0, 1.0).numpy() * self.config['act_scale'] + self.config['act_offset']
        return action

    def get_action_and_log_prob(self, pi):
        if not self.deterministic:
            pi_dist = self.get_distribution(pi)
            action = pi_dist.sample()
            if self.config['bound_corr']:
                below = pi_dist.cdf(action.clamp(-1.0, 1.0)).clamp(self.epsilon, 1.0-self.epsilon).log()
                where_below = action.le(-1).float()
                above = (torch.ones_like(action, device=device) -
                         pi_dist.cdf(action.clamp(-1.0, 1.0))).clamp(self.epsilon, 1.0-self.epsilon).log()
                where_above = action.ge(1).float()
                in_bounds = pi_dist.log_prob(action)
                where_in_bounds = action.gt(-1).float() * action.lt(1).float()
                log_prob = torch.sum(in_bounds * where_in_bounds + below * where_below + above * where_above, dim=-1)
            else:
                log_prob = torch.sum(pi_dist.log_prob(action), dim=-1)
        else:
            log_prob = None  # Not needed for testing
            action = pi[0]
        rescaled_action = (action.clamp(-1.0, 1.0) * self.config['act_scale']
                           + torch.ones_like(action) * self.config['act_offset'])
        return rescaled_action, log_prob


class OffPolicyActorCriticTanhSampler(BaseSampler):

    def __init__(self, config, deterministic):
        super().__init__(config, deterministic)
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)
        self.config.setdefault('min_mu', None)  # CQL version of rlkit clamps to [-9, 9]
        self.config.setdefault('max_mu', None)

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Normal(pi[0], pi[1])

    def get_action(self, pi, random=False, deterministic=None):
        if random:
            return ((2*np.random.rand(*pi[0].cpu().numpy().shape) - 1) * self.config['act_scale'] +
                    self.config['act_offset'])
        if deterministic is None:
            deterministic = self.deterministic
        if not deterministic:
            pi_distribution = self.get_distribution(pi)
            action = pi_distribution.sample()  # don't need reparameterization trick
            action = torch.tanh(action).cpu().numpy() * self.config['act_scale'] + self.config['act_offset']
        else:
            action = torch.tanh(pi[0]).cpu().numpy() * self.config['act_scale'] + self.config['act_offset']
        return action

    def get_action_and_log_prob(self, pi, raw=False, reparam=False):
        if not self.deterministic:
            pi_distribution = self.get_distribution(pi)
            if reparam:
                action = pi_distribution.rsample()
            else:
                action = pi_distribution.sample()
            if self.config.max_mu and self.config.min_mu:
                action = torch.clamp(action, self.config.min_mu, self.config.max_mu)
            log_prob = pi_distribution.log_prob(action).sum(dim=-1)
            if not raw:
                log_prob -= (2 * (np.log(2) - action -
                                  torch.nn.functional.softplus(-2 * action))).sum(axis=-1)
                action = torch.tanh(action) * self.config['act_scale'] + self.config['act_offset']
        else:
            action = torch.tanh(pi[0]).numpy() * self.config['act_scale'] + self.config['act_offset']
            log_prob = -torch.ones((1,), device=device)  # Not needed for testing
        return action.squeeze(0), log_prob.squeeze(0)

    def get_raw_log_prob(self, pi, raw_act):
        """  Assumes a raw action (i.e., unmodified from that sampled by policy)  """
        pi_distribution = self.get_distribution(pi)
        return pi_distribution.log_prob(raw_act).sum(dim=-1).squeeze(0)

    def correct_raw_act(self, raw_action):
        return torch.tanh(raw_action) * self.config['act_scale'] + self.config['act_offset']

    @staticmethod
    def correct_raw_log_prob_pi(raw_log_prob, raw_act):
        correction = (2 * (np.log(2) - raw_act - torch.nn.functional.softplus(-2 * raw_act))).sum(axis=-1).squeeze(0)
        return raw_log_prob - correction


'''
class SafetyEditorSampler(object):
    """  todo this is not yet complete has not been tested!  """
    def __init__(self, config, um, se):
        self.epsilon = 1.e-6
        self.config = config
        self.config.setdefault('act_scale', 1)
        self.config.setdefault('act_offset', 0)
        self.um = um  # network object
        self.se = se  # network object

    @staticmethod
    def get_distribution(pi):
        return torch.distributions.Normal(pi[0], pi[1])

    def h(self, a_hat, delta_a):
        """  Editing function h  """
        return torch.clip(a_hat + 2*delta_a, -self.config.act_scale, self.config.act_scale)

    def get_action(self, obs, random=False, deterministic=False):
        """  Get edited action  """
        if random:
            return ((2 * np.random.rand(self.config.action_dim) - 1) * self.config['act_scale']
                    + self.config['act_offset'])
        if not deterministic:
            pi_um = self.um(obs)
            um_dist = self.get_distribution(pi_um)
            a_um = um_dist.sample()  # don't need reparameterization trick
            a_um = torch.tanh(a_um) * self.config['act_scale'] + self.config['act_offset']  # todo: check tanh used
            pi_se = self.se(torch.cat((obs, a_um), dim=-1))
            se_dist = self.get_distribution(pi_se)
            da_se = se_dist.sample()  # don't need reparameterization trick
            da_se = torch.tanh(da_se) * self.config['act_scale'] + self.config['act_offset']  # todo: check tanh used
            action = self.h(a_um, da_se)
        else:
            pi_um = self.um(obs)
            a_um = torch.tanh(pi_um[0]) * self.config['act_scale'] + self.config['act_offset']  # todo: check tanh used
            pi_se = self.se(torch.cat((obs, a_um), dim=-1))
            da_se = torch.tanh(pi_se[0]) * self.config['act_scale'] + self.config['act_offset']  # todo: check tanh used
            action = self.h(a_um, da_se)
        return action

    def get_action_and_log_prob(self, obs, rsample=0):
        # todo: need to complete, check.  have not thought about this very carefully.
        # idea with the rsample variable is that 0 is no rsample, 1 does it for UM, 2 for SE (corresponding to
        # different updates)
        # note: deterministic mode not needed here
        pi_um = self.um(obs)
        um_dist = self.get_distribution(pi_um)
        if rsample == 1:
            a_um = um_dist.rsample()  # for um update
        else:
            a_um = um_dist.sample()
        log_prob_um = um_dist.log_prob(a_um).sum(dim=-1)
        # I *think* tanh is used on output of both UM and SE; writing this as if it is.  More sure of the former
        # than the latter.
        squash_um = (2 * (np.log(2) - a_um - torch.nn.functional.softplus(-2 * a_um))).sum(dim=-1)
        log_prob_um -= squash_um
        a_um = torch.tanh(a_um) * self.config['act_scale'] + self.config['act_offset']
        pi_se = self.se(torch.cat((obs, a_um), dim=-1))
        se_dist = self.get_distribution(pi_se)
        if rsample == 2:
            da_se = se_dist.rsample()  # for se update
        else:
            da_se = se_dist.sample()
        log_prob_se = um_dist.log_prob(da_se).sum(dim=-1)
        squash_se = (2 * (np.log(2) - da_se - torch.nn.functional.softplus(-2 * da_se))).sum(dim=-1)
        log_prob_se += squash_se
        da_se = torch.tanh(da_se) * self.config['act_scale'] + self.config['act_offset']
        action = self.h(a_um, da_se)
        log_prob = log_prob_um + log_prob_se
        return action, log_prob
'''
