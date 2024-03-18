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
ENVS_NUM = 10 
TRAIN_MODEL_STEP = 1 
TQDM_STEP = 100  
TENSORBOARD_WRITE_STEP = 1000

SAVE_MODEL_STEP = 10000  
EPS_CLIP_DECAY_STEP = 20000 

TIMESTEP = 1000000  

TARGET_UPDATE_STEP = 100  
BATCH_SIZE = 1024 
LEARNING_RATE = 1e-4  
GAMMA_RATE = 0.99  
START_EPS_RATE = 0.1  
EPS_DECREASE_RATIO = 0.95 
FINAL_EPS_RATE = 0.01  
TAU = 0.5  

WEIGHT_DECAY = 1e-8  

MONITOR_DIR = "result/monitor/"+str(time.time_ns())
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())

'''
the setting that you need to load from the game
'''

from game.test_game import CartPoleEnv
game_class = CartPoleEnv
SEQ_LEN = 1
OBS_SHAPE = {
    "box_observation": (SEQ_LEN, 4),
}
ACTION_NUM = 2

from model.test_model_arch import TestModelArch
MODEL_CLASS = TestModelArch

from dqn.replay_buffer import ReplayBuffer
BUFFER_ARCH = ReplayBuffer
BUFFER_SIZE = 50000


import torch as th

OPTIMIZER = th.optim.Adam
LOSS_FUNC = th.nn.SmoothL1Loss
DEVICE = 'cuda'


