import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT            

'''
the basic setting
'''
NEED_TRAIN = True
NEED_RENDER = False                                                                                                                                                                                                                         

'''
the basic model train setting
'''
ENVS_NUM = 10 
TRAIN_MODEL_STEP = 4 
TQDM_STEP = 100  
TENSORBOARD_WRITE_STEP = 1000

SAVE_MODEL_STEP = 10000  
EPS_CLIP_DECAY_STEP = 20000 

TIMESTEP = 1000000  

TARGET_UPDATE_STEP = 400  
BATCH_SIZE = 1024 
LEARNING_RATE = 1e-4  
GAMMA_RATE = 0.99  
START_EPS_RATE = 0.1  
EPS_DECREASE_RATIO = 0.95 
FINAL_EPS_RATE = 0.001  
TAU = 0.5  

WEIGHT_DECAY = 1e-8  

MONITOR_DIR = "result/monitor/"+str(time.time_ns())
MODEL_SAVE_DIR = "result/model"
TENSORBOARD_SAVE_DIR = "result/log/"+str(time.time_ns())

'''
the setting that you need to load from the game
'''

from game.mario_game import mario_env
game_class = mario_env
ACTION_NUM = len(COMPLEX_MOVEMENT)
SEQ_LEN = 4
OBS_SHAPE = {
    'image': (SEQ_LEN, 100, 100),
    "tick": (SEQ_LEN, 1),
    "last_press": (SEQ_LEN, ACTION_NUM)
}


from model.mario_model_arch import DuelingDqnNet
MODEL_CLASS = DuelingDqnNet

from dqn.replay_buffer import ReplayBuffer
BUFFER_ARCH = ReplayBuffer
BUFFER_SIZE = 50000


import torch as th

OPTIMIZER = th.optim.Adam
LOSS_FUNC = th.nn.SmoothL1Loss
DEVICE = 'cuda'


