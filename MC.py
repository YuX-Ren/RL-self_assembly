import numpy as np
import math
from matplotlib import pyplot as plt
def Gaussain_energy(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-((x - mu_x) ** 2 / sigma_x ** 2 + (y - mu_y) ** 2 / sigma_y ** 2))

def V(x,y,bounds = [50,50]):
    mask = (x < 0) | (x > bounds[0]) | (y < 0) | (y > bounds[1])
    mask = np.where(mask, np.inf, 0)
    r = 0
    cos = -50*np.cos(np.sqrt(((x-25)/1.7)**2+((y-25)/1.7)**2))
    trap_0 = -50 * Gaussain_energy(x, y, 25, 25, 4, 4)
    trap_1 = -50 * Gaussain_energy(x, y, 20, 20, 4, 4)
    trap_2 = -30 * Gaussain_energy(x, y, 30, 30, 8, 3)
    trap_3 = -70 * Gaussain_energy(x, y, 7, 25, 4, 15)
    trap_4 = -40 * Gaussain_energy(x, y, 25, 7, 15, 4)
    trap_5 = -50 * Gaussain_energy(x, y, 40, 40, 3, 3)
    return r + cos + trap_0 + trap_1 + trap_2 + trap_3 + trap_4 + trap_5 + mask

def draw_potential(V):
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    X, Y = np.meshgrid(x, y)
    Z = V(X, Y)
    fig = plt.figure()
    # draw a 2D plot
    # denote the value of colortable

    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.savefig('potential.png')

draw_potential(V)

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
    p_1 = np.min(1,np.exp(-V_1_0/T)) + p_0
    p_2 = np.min(1,np.exp(-V1_0/T)) + p_1
    p_3 = np.min(1,np.exp(-V_0_1/T)) + p_2
    p_4 = np.min(1,np.exp(-V0_1/T)) + p_3
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

def mc_simulation(V,T,seed):
    print('T = ',T)
    np.random.seed(seed)
    map_count = np.zeros((50,50))
    x = np.floor(np.random.uniform(0,1)*50)
    y = np.floor(np.random.uniform(0,1)*50)
    for i in range(1000000):
        while x < 0 or x > 49 or y < 0 or y > 49:
            x,y = mc_step(x,y,V,T)
        map_count[int(x),int(y)] += 1
    return map_count
# set mutltiprocess to 24
# pro_num = 100
# from multiprocessing import Pool
# with Pool(processes = pro_num) as pool:
#     results = pool.starmap(mc_simulation, [(V, 300,i) for i in range(pro_num)])
# pool.close()
# pool.join()
# map_count = np.sum(results, axis=0)


# plt.imshow(map_count)
# plt.colorbar()
# plt.savefig('mc_simulation.png')