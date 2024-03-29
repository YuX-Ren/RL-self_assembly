import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import cv2
import csv
from copy import deepcopy

class FlapperEnv():
    def __init__(self, worker_id, frame_tick = 1, num_tick = 4, monitor_file_path = None):
        self.game = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
        self.num_tick = num_tick
        self.frame_tick = frame_tick
        self.state = {
            'image': np.zeros((self.num_tick, 100, 100), dtype=np.float32),
            "tick": np.zeros((self.num_tick, 1), dtype = np.float32),
            "last_press": np.zeros((self.num_tick, self.game.action_space.n), dtype = np.int32)   
        }
        self.action_num = self.game.action_space.n
        self.monitor_file_path = monitor_file_path
        self.ticks = 0.0
        self.all_reward = 0.0
    def reset(self, seed = None):
        
        __, info = self.game.reset()
        
        state = self.game.render()
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (100, 100))
        
        self.state['image'] = np.zeros((self.num_tick, 100, 100), dtype=np.float32)
        self.state['image'][-1] = np.array(state, dtype=np.float32)/255.0
        with open(self.monitor_file_path, 'a', newline='') as fp:
            monitor_file = csv.writer(fp)
            monitor_file.writerow([int(self.ticks), float(self.all_reward)])
        self.ticks = 0.0
        
        self.state['tick'] = np.zeros((self.num_tick, 1), dtype = np.float32)
        self.state["last_press"] = np.zeros((self.num_tick, self.game.action_space.n), dtype = np.float32)
        
        self.all_reward = 0.0
        
        return deepcopy(self.state), info
    
    def step(self, action_index):
        
        all_reward = 0.0
        for i in range(self.frame_tick):
            __, reward, done, truncated, info = self.game.step(action_index)
            state = self.game.render()
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (100, 100))
            all_reward += reward
            self.ticks += 1.0
            if self.ticks >= 200000:
                truncated = True
                done = True
            if done == True:
                break
        self.all_reward += all_reward
        
        self.state['image'][0:-1] = self.state['image'][1:]
        self.state['image'][-1] = np.array(state, dtype=np.float32)/255.0
        self.state['tick'][0:-1] = self.state['tick'][1:]
        self.state['tick'][-1] = self.ticks/2000
        self.state["last_press"][0:-1] = self.state["last_press"][1:]
        self.state["last_press"][-1, :] = 0.0
        self.state["last_press"][-1, action_index] = 1.0
        
        return deepcopy(self.state), all_reward, done, truncated, info
    
    def reset_step(self, action_index):
        
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        
        return return_state, all_reward, done, truncated, info
        
    def render(self):
        # cv2.imshow("render_image", self.state['image'][-1])
        # save the image to the monitor file
        cv2.imwrite(self.monitor_file_path + "show.png", self.state['image'][-1]*255.0)
        cv2.waitKey(10)

if __name__ == '__main__':
    env = FlapperEnv()
    done = True
    frame = 0
    while True:
        if done == True:
            env.reset()
        action = env.action_space.sample()
        print(action)
        state, reward, done, truncated, info = env.step(action)
        frame +=1
        env.render()
