'''Ei ¼ ð0; 0.4; 1; 0.2Þ, and B12 ¼ 1.5,
B13 ¼ 1.1, B23 ¼ 10, B24 ¼ 0.01, B34 ¼ 1, and B14 ¼ ∞
'''

import numpy as np
from scipy.linalg import solve
def rate_cal(B, E, T):
    return np.exp(-(B-E)/T)


class discrete_T_env():
    def __init__(self, seed=0, temp_sele=3, state_num = 4, threshold = 0.01):
        self.state_num = state_num
        self.action_num = temp_sele
        # set the seed
        np.random.seed(seed)
        E = np.array([0, 0.4, 1, 0.2])
        B = np.array([[np.inf, 1.5, 1.1, np.inf], [1.5, np.inf, 10, 0.01], [1.1, 10, np.inf, 1], [np.inf, 0.01, 1, np.inf]])
        self.rate_matrix = np.zeros((self.action_num, self.state_num, self.state_num))
        self.rate_matrix[0] = rate_cal(B, E, 0.1)
        self.rate_matrix[1] = rate_cal(B, E, 0.5)
        self.rate_matrix[2] = rate_cal(B, E, 2)
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
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix[-1])
        eigenvectors = np.transpose(eigenvectors)
        targetvector = eigenvectors[np.argmax(eigenvalues)]
        targetvector = targetvector / np.sum(targetvector)
        self.target_state = targetvector
        print(targetvector)
        print(self.state)

    def calculate_motion_mode(self):
        #decompose the rate_matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.rate_matrix[-1])
        order = np.argsort(np.abs(eigenvalues))
        self.rate_eigenvec = eigenvectors
        # print(eigenvalues)
        # solve the Ax = b
        mask = np.abs(eigenvalues) > 1e-5
        # use A.T * A * x = A.T * b to solve the x
        A = eigenvectors*mask
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

    def step(self, action_index, threshold = -30, time = 0.01):
        # calculate the next state
        done = False
        truncated = False
        reward = 0
        cur_distance = np.log(np.sum(self.target_state *(np.log(self.target_state) - np.log(self.state))))
        # discard the min eigenvalue

        cur_modes, order = self.calculate_motion_mode()
        mask = np.zeros(self.state_num-1, dtype=bool)
        mask[order[-1]] = True
        cur_modes = cur_modes[mask]
        
        rate_matrix = self.rate_matrix[action_index]
        delta_state = np.dot(rate_matrix, self.state) * time
        self.state += delta_state
        # calculate the distance and judge the done
        distance = np.log(np.sum(self.target_state *(np.log(self.target_state) - np.log(self.state))))
        delta_distance = distance - cur_distance
        # reward 
        modes, order = self.calculate_motion_mode()
        # print(modes)
        mask = np.zeros(self.state_num-1, dtype=bool)
        mask[order[-1]] = True
        modes = modes[mask]
        delta_modes = np.log(np.dot(modes, modes)) - np.log(np.dot(cur_modes, cur_modes))
        # if delta_modes < 0:
        #     print(delta_modes)
        reward = -delta_modes - delta_distance
        # print(reward, -delta_modes*10, -delta_distance*10)
        # print(-np.dot(modes, modes), -distance/10)
        # reward = -distance
        if distance < threshold:
            done = True
            reward = 100
        
        return self.state, reward, done, truncated, distance
    
    
    def reset_step(self, action_index):
        
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()

        return return_state, all_reward, done, truncated, info


class continus_T_env():
    def __init__(self, seed=0, boundary = [1,2], state_num = 4, threshold = 0.01):
        self.state_num = state_num
        # set the seed
        np.random.seed(seed)
        self.E = np.array([0, 0.4, 1, 0.2])
        self.B = np.array([[np.inf, 1.5, 1.1, np.inf], [1.5, np.inf, 10, 0.01], [1.1, 10, np.inf, 1], [np.inf, 0.01, 1, np.inf]])
        self.bounds = boundary
        self.init_rate_matrix = rate_cal(self.B, self.E, 1)
        self.end_rate_matrix = rate_cal(self.B, self.E, 2)
        for j in range(self.state_num):
            self.init_rate_matrix[j][j] = - np.sum(self.init_rate_matrix[:,j])
            self.end_rate_matrix[j][j] = - np.sum(self.end_rate_matrix[:,j])

        # nomanlize
        self.init_matrix = self.init_rate_matrix / np.max(np.abs(self.init_rate_matrix)) + np.eye(self.state_num)
        self.end_matrix = self.end_rate_matrix / np.max(np.abs(self.end_rate_matrix)) + np.eye(self.state_num)

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
        eigenvalues, eigenvectors = np.linalg.eig(self.end_rate_matrix)
        self.rate_eigenvec = eigenvectors

        # solve the Ax = b
        mask = np.abs(eigenvalues) > 1e-5
        # use A.T * A * x = A.T * b to solve the x
        A = eigenvectors*mask
        b = self.state-self.target_state
        x = solve(np.dot(A.T, A)[1:,1:], np.dot(A.T, b)[1:])
        return x
    
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
        # regular the temp
        temp = (temp+1)/2 * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        rate_matrix = rate_cal(self.B, self.E, temp)
        for j in range(self.state_num):
            rate_matrix[j][j] = - np.sum(rate_matrix[:,j])
        delta_state = np.dot(rate_matrix, self.state) * time
        self.state += delta_state
        if any(self.state < 0):
            truncated = True
            return self.state, -100, done, truncated, 0
            
        # calculate the distance and judge the done
        distance = np.log(np.sum(self.target_state *(np.log(self.target_state) - np.log(self.state))))
        # for test 
        ########################################################
        reward = -distance/100
        if distance < threshold:
            done = True
            reward = 100
        
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
            if step < 2000:
                state, reward, done, truncated, distance = env.step(1)
            else:
                state, reward, done, truncated, distance = env.step(-1)
            print( reward, done, truncated, distance, step, file=open("test.txt", "a"))
    elif test == 'continus':
        env = continus_T_env()
        done = False
        step = 0
        while not done:
            step = step + 1
            tem = np.random.uniform(1,2)
            state, reward, done, truncated, distance = env.step(tem)

            print( reward, done, truncated, distance, step, file=open("test.txt", "a"))