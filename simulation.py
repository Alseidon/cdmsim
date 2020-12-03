# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt

from time_handler import simulator

N = 2
L = 20
m = 10
"""
t = np.logspace(-1, -0.5, 50)
a = np.sqrt(t)

pos_init = random((N, 3)) * L
mom_init = (random((N, 3)) - 0.5) * 1
#mom_init = np.zeros((N,3))

l_pos = simulator(L, m, a, pos_init, mom_init, show=True)


"""
from motion_solver import mom_update, pos_update
from time_handler import density_update, affiche
from poisson_solver import psolve
def step2b(grid, pos, mom, l, m, a, da):
    _ = mom_update(-grid, pos, mom, a-da/2, da, l)
    _ = pos_update(pos, mom, a, da, l)
    _ = density_update(grid, pos, l, m, delta=False)
    grid = psolve(grid, a)
    return grid

def simulator2b(l, m, a, da, pos_init, mom_init, show=True):
    grid = np.zeros((l,l,l))
    pos, mom = pos_init, mom_init

    line = affiche(pos, 0, l, show, True)
    x_pre = np.copy(pos[0])
    l_pos = [np.copy(pos_init)[0]]
    for current_a in a[1:]:
        grid = step2b(grid, pos, mom, l, m, current_a, da)
        if show:
            # print('a :{:.3f} / {:.3f}--- p : {} --- dx : {}'.format(current_a, a[-1], mom[0], pos[0]- x_pre))
            print('E_tot : {}'.format(np.sum([np.sum(pos[i]**2) for i in range(len(pos))]) / (2 * m) ))
        x_pre = np.copy(pos[0])
        l_pos.append(np.copy(pos[0]))
        affiche(pos, line, l, show, False)
        plt.pause(0.01)
    affiche(pos, line, l, show, False)
    plt.show()
    return l_pos

pos_init = np.array([[L/2 +i, L/2 +i, L/2 +i] for i in range(N)])
mom_init = np.zeros((N, 3))
mom_init[0, 0] = 1
# mom_init[0, 0], mom_init[1, 0] = 0.5, -0.5

a = np.ones(50) * 1
da = 0.05

lpos = simulator2b(L, m, a, da, pos_init, mom_init, True)
_ = input("Next ? ")
x, y, z = [], [], []
for point in lpos :
    x.append(point[0])# %% [markdown]
    y.append(point[1])
    z.append(point[2])
x, y, z = np.array(x), np.array(y), np.array(z)
plt.figure()
plt.plot((x[1:] - x[:-1]) / da, 'r.', label = 'v_x')
plt.plot((y[1:] - y[:-1]) / da, '.', label = 'v_y')
#plt.plot((z[1:] - z[:-1]) / da, 'g.', label = 'v_z')
plt.legend()
plt.show()
plt.figure()
plt.plot(x, label='x.')
plt.plot(y, label='y.')
plt.plot(z, label='z.')
plt.legend()
plt.show()