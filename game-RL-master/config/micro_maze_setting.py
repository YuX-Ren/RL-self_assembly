import os
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

            

'''
the basic setting
'''
NEED_TRAIN = True
NEED_RENDER =False
'''
the basic model train setting
'''
ENVS_NUM = 100 
TRAIN_MODEL_STEP = 1 
TQDM_STEP = 100   
TENSORBOARD_WRITE_STEP = 1000

SAVE_MODEL_STEP = 10000  
EPS_CLIP_DECAY_STEP = 20000 

TIMESTEP = 1000000  

TARGET_UPDATE_STEP = 100  
BATCH_SIZE = 1024 
LEARNING_RATE = 1e-4  
GAMMA_RATE = 0.8
START_EPS_RATE = 0.5  
EPS_DECREASE_RATIO = 0.95 
FINAL_EPS_RATE = 0.01  
TAU = 1

WEIGHT_DECAY = 1e-8  

MONITOR_DIR = "result/monitor/"+str(time.time_ns())
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())

'''
the setting that you need to load from the game
'''

from game.micro_maze import Discret_env
game_class = Discret_env
SEQ_LEN = 1
OBS_SHAPE = {
    "site": (SEQ_LEN, 2),
    "image": (5, 5)
}
ACTION_NUM = 3

from model.micro_maze_arch import DuelingDqnNet
MODEL_CLASS = DuelingDqnNet

from dqn.replay_buffer import ReplayBuffer
BUFFER_ARCH = ReplayBuffer
BUFFER_SIZE = 50000


import torch as th

OPTIMIZER = th.optim.Adam
LOSS_FUNC = th.nn.SmoothL1Loss
DEVICE = 'cuda'


