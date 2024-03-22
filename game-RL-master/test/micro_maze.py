import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.micro_maze import Discret_env
from model.micro_maze_arch import DuelingDqnNet
import os
import torch as th
if __name__ == '__main__':
    env = Discret_env(0,frame_tick=1)
    done = False
    SEQ_LEN = 1
    OBS_SHAPE = {
        "site": (SEQ_LEN, 2),
        "image": (5, 5)
    }
    ACTION_NUM = 3


    model = DuelingDqnNet(OBS_SHAPE, ACTION_NUM)
    # load the model
    model_file_dir = "result/model"
    load_file_name = '0'
    for file_name in os.listdir(model_file_dir):
        if file_name.endswith(".pt") and file_name > load_file_name:
            load_file_name = file_name

    if load_file_name != '0':
        load_file_path = model_file_dir + '/'+load_file_name
        print("found the newest file %s" % load_file_path, " try to load")
        model.load_state_dict(th.load(load_file_path))
    state, _ = env.reset()
    for key, value in state.items():
        state[key] = th.FloatTensor(value)
    step = 0
    while not done:
        step +=1
        if step < 20000:
            action = model(state).argmax().numpy()
            state, reward, done, truncated, info = env.step(action)
            print(reward, done, action, info, step)
        # else:
        #     # break
        #     action = model(state).argmax().numpy()
        #     state, reward, done, truncated, info = env.step(2)
        # print(reward, done, action, info, step, file=open("test.txt", "a"))
        for key, value in state.items():
            state[key] = th.FloatTensor(value)
    print(reward, done, action, info, step)
    