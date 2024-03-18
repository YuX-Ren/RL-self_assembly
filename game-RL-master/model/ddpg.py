import torch as th
from torch import nn
from torch.nn import functional as F
'''
TODO
init_weights
__init__ compatible with the setting
'''
class actor(nn.Module):
    def __init__(self, obs_shape_dict, hidden_dim):
        super(actor, self).__init__()

        self.hidden_dim = hidden_dim
        self.state_dim = obs_shape_dict["Mstate"][1]
        self.layer1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, 1)
        self.out = nn.Tanh()

    def forward(self, observations) -> th.Tensor:
        state = observations["Mstate"].squeeze(dim = -2)
        out = F.relu(self.layer1(state))
        out = F.relu(self.layer2(out))
        out = self.out(self.layer3(out))
        return out

class critic(nn.Module):
    def __init__(self, obs_shape_dict, hidden_dim):
        super(critic, self).__init__()
        self.state_dim = obs_shape_dict["Mstate"][1]
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim+1, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, observations, action) -> th.Tensor:
        state = observations["Mstate"].squeeze(dim = -2)
        out = F.relu(self.layer1(state))
        out = th.cat([out, action], dim = 1)
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out



import numpy as np 

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma
    
    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma
    
    
class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta=0.5, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)