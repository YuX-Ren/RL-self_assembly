import numpy as np
from scipy.linalg import solve
def rate_cal(B, E, T):
    return np.exp(-(B-E)/T)

def Gaussain_energy(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-((x - mu_x) ** 2 / sigma_x ** 2 + (y - mu_y) ** 2 / sigma_y ** 2))

def V(x,y,bounds = [50,50]):
    mask = (x < 0) | (x > bounds[0]) | (y < 0) | (y > bounds[1])
    mask = np.where(mask, np.inf, 0)
    r = np.sqrt((x-25)**2 + (y-25)**2)
    cos = -80*np.cos(np.sqrt(((x-25)/1.7)**2+((y-25)/1.7)**2))
    trap_0 = -50 * Gaussain_energy(x, y, 25, 25, 4, 4)
    trap_1 = -50 * Gaussain_energy(x, y, 20, 20, 4, 4)
    trap_2 = -30 * Gaussain_energy(x, y, 30, 30, 8, 3)
    trap_3 = -70 * Gaussain_energy(x, y, 7, 25, 4, 15)
    trap_4 = -40 * Gaussain_energy(x, y, 25, 7, 15, 4)
    trap_5 = -50 * Gaussain_energy(x, y, 40, 40, 3, 3)
    return r + cos + trap_0 + trap_1 + trap_2 + trap_3 + trap_4 + trap_5 + mask

class Micro_discret_env():
    def __init__(self, seed=0, temp_sele=3, threshold = 0.01, boundary = [280,350],potential_func=V,target_state = (25, 25)):
        self.action_num = temp_sele
        self.height = boundary[1]
        self.low = boundary[0]
        self.potential_func = potential_func
        x,y = np.meshgrid(np.arange(50),np.arange(50))
        self.potential_image = self.potential_func(x,y).T
        # set the seed
        # np.random.seed(seed)
        self.state = self.reset()
        self.target_state = target_state
    
    @staticmethod
    def mc_step(x,y,V,T):
        # calculate the value of the potential
        V0 = V(x,y)
        # calculate the value of the potential after the discrete move
        V_1_0 = V(x-1,y) - V0
        V1_0 = V(x+1,y) - V0
        V_0_1 = V(x,y-1) - V0
        V0_1 = V(x,y+1) - V0
        # generate the judge boundary
        p_0 = 1
        p_1 = min(1,np.exp(-V_1_0/T)) + p_0
        p_2 = min(1,np.exp(-V1_0/T)) + p_1
        p_3 = min(1,np.exp(-V_0_1/T)) + p_2
        p_4 = min(1,np.exp(-V0_1/T)) + p_3
        # judge the direction
        z_inv = p_4
        # generate a random number
        r = np.random.uniform(0,1) 
        print(min(1,np.exp(-V_1_0/T)),min(1,np.exp(-V1_0/T)),min(1,np.exp(-V_0_1/T)),min(1,np.exp(-V0_1/T)))
        # print(r, 1/z_inv,p_1/z_inv, p_2/z_inv, p_3/z_inv, p_4/z_inv)
        # the special case delat_P = 0 is considered
        if r < 1/z_inv:
            return x,y
        elif r < p_1/z_inv:
            return x-1,y
        elif r < p_2/z_inv:
            return x+1,y
        elif r < p_3/z_inv:
            return x,y-1
        else:
            return x,y+1

    def reset(self):
        # return 44, 48
        return np.random.randint(0,50), np.random.randint(0,50)

    def step(self, action_index, threshold = -30, time = 0.001):
        # calculate the next state
        done = False
        truncated = False
        reward = 0
        T = action_index/self.action_num *(self.height - self.low) + self.low
        pre_distance = np.sqrt((self.state[0] - self.target_state[0])**2 + (self.state[1] - self.target_state[1])**2)
        x, y = self.state
        self.state = self.mc_step(x,y,self.potential_func,T)
        if self.state == self.target_state:
            done = True
            reward = 10
            return self.state, reward, done, truncated, 0
        distance = np.sqrt((self.state[0] - self.target_state[0])**2 + (self.state[1] - self.target_state[1])**2)
        reward = -0.01 
        return self.state, reward, done, truncated, distance
    
    
    def reset_step(self, action_index):
        
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()

        return return_state, all_reward, done, truncated, info

class Micro_continus_env():
    def __init__(self, seed=0, threshold = 0.01, boundary = [280,320],potential_func=V,target_state = (25,25)):
        self.height = boundary[1]
        self.low = boundary[0]
        self.potential_func = potential_func
        # set the seed
        np.random.seed(seed)
        self.state = self.reset()
        self.target_state = target_state

    @staticmethod
    def mc_step(x,y,V,T):
        # calculate the value of the potential
        V0 = V(x,y)
        # calculate the value of the potential after the discrete move
        V_1_0 = V(x-1,y) - V0
        V1_0 = V(x+1,y) - V0
        V_0_1 = V(x,y-1) - V0
        V0_1 = V(x,y+1) - V0
        # generate the judge boundary
        p_0 = 1
        p_1 = min(1,np.exp(-V_1_0/T)) + p_0
        p_2 = min(1,np.exp(-V1_0/T)) + p_1
        p_3 = min(1,np.exp(-V_0_1/T)) + p_2
        p_4 = min(1,np.exp(-V0_1/T)) + p_3
        # judge the direction
        z_inv = p_4
        # generate a random number
        r = np.random.uniform(0,1) 
        # the special case delat_P = 0 is considered
        if r < 1/z_inv:
            return x,y
        elif r < p_1/z_inv:
            return x-1,y
        elif r < p_2/z_inv:
            return x+1,y
        elif r < p_3/z_inv:
            return x,y-1
        else:
            return x,y+1
        
    def reset(self):
        return np.random.randint(0,50), np.random.randint(0,50)

    def step(self, action, threshold = -30, time = 0.001):
        # calculate the next state
        done = False
        truncated = False
        reward = 0
        T = (action+1)/2*(self.height - self.low) + self.low
        x, y = self.state
        self.state = self.mc_step(x,y,self.potential_func,T)
        if self.state == self.target_state:
            done = True
            reward = 10
            return self.state, reward, done, truncated, 0
        distance = np.sqrt((self.state[0] - self.target_state[0])**2 + (self.state[1] - self.target_state[1])**2)
        reward = -0.1
        return self.state, reward, done, truncated, distance


if __name__ == '__main__':
    test = 'discret'
    if test == 'discret':
        env = Micro_discret_env()
        done = False
        state = env.reset()
        step = 0
        while not done:
            step +=1
            action = 0
            state, reward, done, truncated, info = env.step(action)
            print(reward, done, action, info, state, step)
            # break
    else:
        env = Micro_continus_env()
        done = False
        state = env.reset()
        while not done:
            action = 0
            state, reward, done, truncated, info = env.step(action)
            print(reward, done, action, info, state)