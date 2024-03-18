import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.asm_game import  Discret_Env
from model.asm_game_arch import AsmModelArch
import os
import torch as th
if __name__ == '__main__':
    env = Discret_Env(0,frame_tick=1)
    done = False
    OBS_SHAPE = {
        "Mstate": (1, 4),
    }    
    model = AsmModelArch(OBS_SHAPE,3)
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
        if step < 3:
            action = model(state).argmax().numpy()
            state, reward, done, truncated, info = env.step(2)
        else:
            # break
            action = model(state).argmax().numpy()
            state, reward, done, truncated, info = env.step(2)
        print(reward, done, action, info, step, file=open("test.txt", "a"))
        for key, value in state.items():
            state[key] = th.FloatTensor(value)