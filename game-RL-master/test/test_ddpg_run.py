import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.asm_game import  Continus_Env
from model.ddpg import actor
import os
import torch as th
if __name__ == '__main__':
    env = Continus_Env(0,frame_tick=1)
    done = False
    OBS_SHAPE = {
        "Mstate": (1, 4),
    }    
    model = actor(OBS_SHAPE,256)
    # load the model
    model_file_dir = "result/model"
    cur_time_ = '0'
    for file_name in os.listdir(model_file_dir):
        time_ = file_name.split('_')[1].split('.')[0]
        if file_name.endswith(".pt") and time_ > cur_time_:
            cur_time_ = time_

    if cur_time_ != '0':
        load_file_path = model_file_dir + '/'+cur_time_
        print("found the newest file %s" % load_file_path, " try to load")
        actor_file = model_file_dir + '/actor_%s.pt' % cur_time_
        model.load_state_dict(th.load(actor_file))
    state, _ = env.reset()
    for key, value in state.items():
        state[key] = th.FloatTensor(value)
    step = 0
    while not done:
        if step > 100:
            break
        step +=1
        action = model(state).detach().numpy()
        state, reward, done, truncated, info = env.step(action)
        print(reward, done, action, info, step)
        for key, value in state.items():
            state[key] = th.FloatTensor(value)