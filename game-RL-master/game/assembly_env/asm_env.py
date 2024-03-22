'''Ei ¼ ð0; 0.4; 1; 0.2Þ, and B12 ¼ 1.5,
B13 ¼ 1.1, B23 ¼ 10, B24 ¼ 0.01, B34 ¼ 1, and B14 ¼ ∞
'''

import numpy as np
from scipy.linalg import solve
def rate_cal(B, E, T):
    return np.exp(-(B-E)/T)


class discrete_T_env():
    def __init__(self, seed=0, temp_sele=10, state_num = 4, threshold = 0.01, boundary = [0, 1]):
        self.state_num = state_num
        self.action_num = temp_sele
        self.height = boundary[1]
        self.low = boundary[0]
        # set the seed
        np.random.seed(seed)
        E = np.array([0, 0.4, 1, 0.2])
        B = np.array([[np.inf, 1.5, 1.1, np.inf], [1.5, np.inf, 10, 0.01], [1.1, 10, np.inf, 1], [np.inf, 0.01, 1, np.inf]])
        self.rate_matrix = np.zeros((self.action_num, self.state_num, self.state_num))
        for i in range(self.action_num):
            self.rate_matrix[i] = rate_cal(B, E, (i+1)/self.action_num*(self.height-self.low)+self.low)

        for i in range(self.action_num):
            for j in range(self.state_num):
                self.rate_matrix[i][j][j] = - np.sum(self.rate_matrix[i][:,j])

        # nomanlize
        self.matrix = np.zeros((self.action_num, self.state_num, self.state_num))
        for i in range(self.action_num):
            self.matrix[i] = self.rate_matrix[i] / np.max(np.abs(self.rate_matrix[i])) + np.eye(self.state_num)

        #ini the state
        ini_rate_matrix = rate_cal(B, E, 1)
        for j in range(self.state_num):
            ini_rate_matrix[j][j] = - np.sum(ini_rate_matrix[:,j])
        self.ini_matrix = ini_rate_matrix / np.max(np.abs(ini_rate_matrix)) + np.eye(self.state_num)
        eigenvalues, eigenvectors = np.linalg.eig(self.ini_matrix)
        eigenvectors = np.transpose(eigenvectors)
        eigenvector = eigenvectors[np.argmax(eigenvalues)]
        self.state = eigenvector / np.sum(eigenvector)

        # target state
        self.target_rate_matrix = rate_cal(B, E, 2)
        for j in range(self.state_num):
            self.target_rate_matrix[j][j] = - np.sum(self.target_rate_matrix[:,j])
        eigenvalues, eigenvectors = np.linalg.eig(self.target_rate_matrix)
        eigenvectors = np.transpose(eigenvectors)
        targetvector = eigenvectors[np.argmax(eigenvalues)]
        targetvector = targetvector / np.sum(targetvector)
        self.target_state = targetvector
        print(targetvector)
        print(self.state)

    def calculate_motion_mode(self):
        #decompose the rate_matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.target_rate_matrix)
        self.rate_eigenvec = eigenvectors
        mask = np.abs(eigenvalues) > 1e-5
        order = np.argsort(np.abs(eigenvalues[mask]))
        # use A.T * A * x = A.T * b to solve the x
        A = eigenvectors
        b = self.state-self.target_state
        x = solve(np.dot(A.T, A)[1:,1:], np.dot(A.T, b)[1:])

        return x, order

    def reset(self):
        #ini the state
        eigenvalues, eigenvectors = np.linalg.eig(self.ini_matrix)
        eigenvectors = np.transpose(eigenvectors)
        eigenvector = eigenvectors[np.argmax(eigenvalues)]
        self.state = eigenvector / np.sum(eigenvector)
        return self.state

    def step(self, action_index, threshold = -30, time = 0.001):
        # calculate the next state
        done = False
        truncated = False
        reward = 0
        cur_distance = np.log(np.sum(self.target_state *(np.log(self.target_state) - np.log(self.state))))
        # discard the min eigenvalue

        cur_modes, order = self.calculate_motion_mode()
        mask = np.zeros(self.state_num-1, dtype=bool)
        mask[order[0]] = True
        cur_modes = cur_modes[mask]
        
        rate_matrix = self.rate_matrix[action_index]
        delta_state = np.dot(rate_matrix, self.state) * time
        self.state += delta_state
        # calculate the distance and judge the done
        distance = np.log(np.sum(self.target_state *(np.log(self.target_state) - np.log(self.state))))
        delta_distance = distance - cur_distance
        # reward 
        modes, order = self.calculate_motion_mode()
        mask = np.zeros(self.state_num-1, dtype=bool)
        mask[order[0]] = True
        modes = modes[mask]
        print(modes)
        delta_modes = np.log(np.abs(modes)) - np.log(np.abs(cur_modes))
        # TODO: variant coefficient modes reward
        reward = -delta_modes[0]*10
        # reward = -distance
        if np.log(np.abs(modes)) < threshold:
            print(modes)
            done = True
            reward = 10
        
        return self.state, reward, done, truncated, distance
    
    
    def reset_step(self, action_index):
        
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()

        return return_state, all_reward, done, truncated, info


class continus_T_env():
    def __init__(self, seed=0, boundary = [0.1,1], state_num = 4, threshold = 0.01):
        self.state_num = state_num
        # set the seed
        np.random.seed(seed)
        self.E = np.array([0, 0.4, 1, 0.2])
        self.B = np.array([[np.inf, 1.5, 1.1, np.inf], [1.5, np.inf, 10, 0.01], [1.1, 10, np.inf, 1], [np.inf, 0.01, 1, np.inf]])
        self.height = boundary[1]
        self.low = boundary[0]
        self.init_rate_matrix = rate_cal(self.B, self.E, 1)
        self.target_rate_matrix = rate_cal(self.B, self.E, 2)
        for j in range(self.state_num):
            self.init_rate_matrix[j][j] = - np.sum(self.init_rate_matrix[:,j])
            self.target_rate_matrix[j][j] = - np.sum(self.target_rate_matrix[:,j])

        # nomanlize
        self.init_matrix = self.init_rate_matrix / np.max(np.abs(self.init_rate_matrix)) + np.eye(self.state_num)
        self.end_matrix = self.target_rate_matrix / np.max(np.abs(self.target_rate_matrix)) + np.eye(self.state_num)

        #ini the state
        self.state = self.reset()

        # target state
        eigenvalues, eigenvectors = np.linalg.eig(self.end_matrix)
        eigenvectors = np.transpose(eigenvectors)
        targetvector = eigenvectors[np.argmax(eigenvalues)]
        targetvector = targetvector / np.sum(targetvector)
        self.target_state = targetvector

    def calculate_motion_mode(self):
        #decompose the rate_matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.target_rate_matrix)
        self.rate_eigenvec = eigenvectors
        mask = np.abs(eigenvalues) > 1e-5
        order = np.argsort(np.abs(eigenvalues[mask]))
        # use A.T * A * x = A.T * b to solve the x
        A = eigenvectors
        b = self.state-self.target_state
        x = solve(np.dot(A.T, A)[1:,1:], np.dot(A.T, b)[1:])

        return x, order
    
    def reset(self):
        #ini the state
        eigenvalues, eigenvectors = np.linalg.eig(self.init_matrix)
        eigenvectors = np.transpose(eigenvectors)
        eigenvector = eigenvectors[np.argmax(eigenvalues)]
        self.state = eigenvector / np.sum(eigenvector)
        return self.state

    def step(self, temp, threshold = -40, time = 0.001):
        # calculate the next state
        done = False
        truncated = False
        reward = 0
        # calculate the current motion mode
        cur_modes, order = self.calculate_motion_mode()
        mask = np.zeros(self.state_num-1, dtype=bool)
        mask[order[0]] = True
        cur_modes = cur_modes[mask]
        # regular the temp
        temp = (temp+1)/2 * (self.height - self.low) + self.low
        temp = 0.1
        rate_matrix = rate_cal(self.B, self.E, temp)
        for j in range(self.state_num):
            rate_matrix[j][j] = - np.sum(rate_matrix[:,j])
        delta_state = np.dot(rate_matrix, self.state) * time
        self.state += delta_state
        # calculate the modes and reward
        modes, order = self.calculate_motion_mode()
        mask = np.zeros(self.state_num-1, dtype=bool)
        mask[order[0]] = True
        modes = modes[mask]
        # print(modes)
        delta_modes = np.log(np.abs(modes)) - np.log(np.abs(cur_modes))
        # TODO: variant coefficient modes reward
        reward = -delta_modes[0]*10
        distance = np.log(np.sum(self.target_state *(np.log(self.target_state) - np.log(self.state))))
        if np.log(np.abs(modes)) < -10:
            print(modes)
            done = True
            reward = 10            
        
        return self.state, reward, done, truncated, distance

    def reset_step(self, temp):
        return_state, all_reward, done, truncated, info = self.step(temp)
        if done == True:
            self.reset()

        return return_state, all_reward, done, truncated, info


if __name__ == '__main__':
    test = 'continus'
    if test == 'discrete':
        env = discrete_T_env()
        done = False
        step = 0
        while not done:
            step = step + 1
            if step < 31:
                state, reward, done, truncated, distance = env.step(0)
                print(reward)
            # else:
            #     state, reward, done, truncated, distance = env.step(-1)
            # print( reward, done, truncated, distance, step, file=open("test.txt", "a"))
    elif test == 'continus':
        env = continus_T_env()
        done = False
        step = 0
        while not done:
            if step == 100:
                break
            step = step + 1
            tem = np.random.uniform(-1,1)
            state, reward, done, truncated, distance = env.step(-1)

            print( reward, done, truncated, distance, step)