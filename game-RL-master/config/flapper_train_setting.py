import os
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

'''
the basic setting
'''
NEED_TRAIN = True
NEED_RENDER = True                                                                                                                                                                                                                         

'''
the basic model train setting
'''
ENVS_NUM = 1 
TRAIN_MODEL_STEP = 4 
TQDM_STEP = 100  
TENSORBOARD_WRITE_STEP = 1000

SAVE_MODEL_STEP = 20000  
EPS_CLIP_DECAY_STEP = 3000 

TIMESTEP = 1000000  

TARGET_UPDATE_STEP = 400  
BATCH_SIZE = 1024 
LEARNING_RATE = 1e-4  
GAMMA_RATE = 0.99  
START_EPS_RATE = 0.005  
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

from game.flapper_game import FlapperEnv
game_class = FlapperEnv
SEQ_LEN = 4
OBS_SHAPE = {
    'image': (SEQ_LEN, 100, 100),
    "tick": (SEQ_LEN, 1),
    "last_press": (SEQ_LEN, 2)
}
ACTION_NUM = 2

from model.flapper_model_arch import DuelingDqnNet
MODEL_CLASS = DuelingDqnNet

from dqn.replay_buffer import ReplayBuffer
BUFFER_ARCH = ReplayBuffer
BUFFER_SIZE = 20000


import torch as th

OPTIMIZER = th.optim.Adam
LOSS_FUNC = th.nn.SmoothL1Loss
DEVICE = 'cuda'
