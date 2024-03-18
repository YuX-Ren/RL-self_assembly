import torch as th
from torch import nn
import time
from copy import deepcopy
from .ddpg import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
class DdpgModelWrapper():
    def __init__(
        self, 
        model_arch:nn.Module, 
        optimizer: th.optim, 
        loss_function: nn, 
        obs_shape_dict: dict, 
        learning_rate: float, 
        gamma: float, 
        weight_decay: float, 
        tau: float,
        device: str, 
        model_save_dir: str):
        self.__device = device
        self.input_shape = obs_shape_dict
        self.random_process = GaussianWhiteNoiseProcess(mu = 0, sigma = 0.2, sigma_min = 0.05, n_steps_annealing = 1000)
        self.actor, self.actor_target, self.critic, self.critic_target = self.__build_model(model_arch, obs_shape_dict)
        self.actor_optim, self.critic_optim = self.__build_optimizer(optimizer, learning_rate, weight_decay)
        self.loss_func =self.__build_loss_function(loss_function)        
        self.gamma = gamma
        self.tau = tau
        self.model_save_dir = model_save_dir
    

    @property
    def device(self):
        return self.__device
    
    @device.setter
    def device(self, device: str):
        self.__device = device
    
    def __build_model(self, model_arch_list:list[nn.Module], obs_shape_dict: dict):
        actor = model_arch_list[0](obs_shape_dict, hidden_dim = 256)
        actor_target = model_arch_list[0](obs_shape_dict, hidden_dim = 256)
        critic = model_arch_list[1](obs_shape_dict, hidden_dim = 256)
        critic_target = model_arch_list[1](obs_shape_dict, hidden_dim = 256)
        return actor.to(self.__device), actor_target.to(self.__device), critic.to(self.__device), critic_target.to(self.__device)
    
    def __build_optimizer(self, optimizer, lr, weight_decay):
        return optimizer(self.actor.parameters(), lr = lr, weight_decay = weight_decay), optimizer(self.critic.parameters(), lr = lr, weight_decay = weight_decay)
    
    def __build_loss_function(self, loss_function):
        return loss_function().to(device = self.__device)
    
    def numpy2net_tensor(self, states_dict: dict)-> dict:
        states = deepcopy(states_dict)
        for key, value in states_dict.items():
            states[key] = th.FloatTensor(value).to(self.__device)
        return states
    
    def train(self, states, actions, rewards, next_states, dones, truncateds):
        # train the critic
        next_q = self.critic_target(next_states, self.actor_target(next_states))
        target_q = rewards + (1 - dones) * self.gamma * next_q
        q = self.critic(states, actions)
        critic_loss = self.loss_func(q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # train the actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the target
        return actor_loss.item() + critic_loss.item()
    
        
    def target_update(self):
        for target_param, evaluation_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * evaluation_param.data + (1 - self.tau) * target_param.data)
        for target_param, evaluation_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * evaluation_param.data + (1 - self.tau) * target_param.data)

    def save(self):
        time_ = str(time.time_ns())
        th.save(self.actor_target.state_dict(),
            self.model_save_dir+"/actor_%s.pt" % (time_))
        th.save(self.critic_target.state_dict(),
            self.model_save_dir+"/critic_%s.pt" % (time_))
    def load(self, file_name:str):
        print("load the model %s" % file_name)
        self.actor.load_state_dict(th.load('actor_' + file_name))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(th.load('critic_'+file_name))
        self.critic_target.load_state_dict(self.critic.state_dict())
    def load_newest(self, model_file_dir: str):
        import os
        cur_time_ = '0'
        for file_name in os.listdir(model_file_dir):
            time_ = file_name.split('_')[1].split('.')[0]
            if file_name.endswith(".pt") and time_ > cur_time_:
                cur_time_ = time_

        if cur_time_ != '0':
            load_file_path = model_file_dir + '/'+cur_time_
            print("found the newest file %s" % load_file_path, " try to load")
            actor_file = model_file_dir + '/actor_%s.pt' % cur_time_
            critic_file = model_file_dir + '/critic_%s.pt' % cur_time_
            self.actor.load_state_dict(th.load(actor_file))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic.load_state_dict(th.load(critic_file))
            self.critic_target.load_state_dict(self.critic.state_dict())
            
def configure_optimizers(model: nn.Module, weight_decay:float):
    # Parameters must have a defined order.
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (th.nn.Linear, th.nn.Conv2d)
    blacklist_weight_modules = (
        th.nn.BatchNorm1d, th.nn.BatchNorm2d, th.nn.LayerNorm, th.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                    # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(
        inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )
    optim_groups = [
        {"params": [param_dict[pn]
                    for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn]
                    for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups