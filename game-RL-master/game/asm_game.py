import gymnasium as gym
import numpy as np
import csv
from copy import deepcopy
from .assembly_env.asm_env import discrete_T_env, continus_T_env
class Discret_Env():
    def __init__(self, worker_id, frame_tick = 1, num_tick = 1,monitor_file_path = None):
        self.num_tick = num_tick
        self.frame_tick = frame_tick
        self.action_num = 10
        self.monitor_file_path = monitor_file_path
        self.state = {
            "Mstate": np.zeros((self.num_tick, 4), dtype = np.int32)   
        }
        self.ticks:int = 0
        self.all_reward = 0.0
        self.game = discrete_T_env(seed=worker_id, temp_sele=self.action_num, state_num = 4, threshold = 0.01)
    def reset(self, seed = None):
        state = self.game.reset()
        for key in self.state.keys():
            if key == "Mstate":
                self.state[key] =  np.zeros((self.num_tick,4), dtype=np.float32)
                self.state[key][-1] = np.array(state, dtype=np.float32)
        self.ticks = 0
        self.all_reward = 0.0
        return deepcopy(self.state) , None
    
    def step(self, action_index):
        all_reward: float = 0.0

        state, reward, done, truncated, info = self.game.step(action_index)
        all_reward += reward
        self.ticks += 1

        self.all_reward += all_reward
        for key in self.state.keys():
            if key == "Mstate":
                self.state[key][0:-1] = self.state[key][1:]
                self.state[key][-1] = np.array(state, dtype=np.float32)
        return deepcopy(self.state), all_reward, done, truncated, info
    def reset_step(self, action_index):
        return_state, step_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        return return_state, step_reward, done, truncated, info

class Continus_Env():
    def __init__(self, worker_id, frame_tick = 1, num_tick = 1,monitor_file_path = None):
        self.num_tick = num_tick
        self.frame_tick = frame_tick
        self.monitor_file_path = monitor_file_path
        self.state = {
            "Mstate": np.zeros((self.num_tick, 4), dtype = np.int32)   
        }
        self.ticks:int = 0
        self.all_reward = 0.0
        self.game = continus_T_env(seed=worker_id, boundary=[0.1,4], state_num = 4, threshold = 0.01)
    def reset(self, seed = None):
        state = self.game.reset()
        for key in self.state.keys():
            if key == "Mstate":
                self.state[key] =  np.zeros((self.num_tick,4), dtype=np.float32)
                self.state[key][-1] = np.array(state, dtype=np.float32)
        self.ticks = 0
        self.all_reward = 0.0
        return deepcopy(self.state) , None
    
    def step(self, action_index):
        all_reward: float = 0.0
        state, reward, done, truncated, info = self.game.step(action_index)
        all_reward += reward
        self.ticks += 1

        self.all_reward += all_reward
        for key in self.state.keys():
            if key == "Mstate":
                self.state[key][0:-1] = self.state[key][1:]
                self.state[key][-1] = np.array(state, dtype=np.float32)
        return deepcopy(self.state), all_reward, done, truncated, info
    def reset_step(self, action_index):
        return_state, step_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        return return_state, step_reward, done, truncated, info
