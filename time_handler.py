# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from motion_solver import mom_update, pos_update
from poisson_solver import psolve

def density_update(grid_dens, pos, l, m, delta=True): #BUGGED
    for i in range(l):
        for j in range(l):
            for k in range(l):
                grid_dens[i,j,k] = 0
    for x in pos:
        i, j, k = np.floor(x)
        i, j, k = int(i), int(j), int(k)
        d = x - np.array([i, j, k])
        t = 1 - d
        grid_dens[i, j, k]             += m * t[0] * t[1] * t[2]
        grid_dens[(i+1)%l, j, k]       += m * d[0] * t[1] * t[2]
        grid_dens[i, (j+1)%l, k]       += m * t[0] * d[1] * t[2]
        grid_dens[i, j, (k+1)%l]       += m * t[0] * t[1] * d[2]
        grid_dens[(i+1)%l, (j+1)%l, k] += m * d[0] * d[1] * t[2]
        grid_dens[(i+1)%l, j, (k+1)%l] += m * d[0] * t[1] * d[2]
        grid_dens[i, (j+1)%l, (k+1)%l] += m * t[0] * d[1] * d[2]
        grid_dens[(i+1)%l, (j+1)%l, (k+1)%l] += m * d[0] * d[1] * d[2]
    if delta: # if we want to work with delta tilde instead of rho
        grid_dens -= np.ones(np.shape(grid_dens))
    return 0



def step(grid, pos, mom, l, m, a, da):
    _ = mom_update(-grid, pos, mom, a, da, l)
    _ = pos_update(pos, mom, l, a, da)
    _ = density_update(grid, pos, l, m)
    _ = psolve(grid, a)
    return



def affiche(pos, line, l, aff, starting):
    if aff:
        if starting:
            line, = plt.plot(pos[:,0], pos[:,1], '.')
            plt.xlim(0, l)
            plt.ylim(0, l)
            print(plt.figaspect(1))
            plt.grid()
        else:
            line.set_xdata(pos[:,0])
            line.set_ydata(pos[:,1])
    return line



def simulator(l, m, a, pos_init, mom_init):
    grid = np.zeros((l,l,l))
    pos, mom = pos_init, mom_init

    line = affiche(pos, 0, l, True, True)

    x_pre = np.copy(pos_init[0])
    l_pos = [np.copy(pos_init)[0]]
    pre_a = a[0]
    for current_a in a[1:]:
        da = current_a - pre_a
        pre_a = current_a
        step(grid, pos, mom, l, m, current_a, da)
        print('a :{:.3f} / {:.3f}--- p : {} --- dx : {}'.format(current_a, a[-1], mom[0], pos[0]- x_pre))
        x_pre = np.copy(pos[0])
        l_pos.append(np.copy(pos[0]))
        affiche(pos, line, l, True, False)
        plt.pause(0.01)
    affiche(pos, line, l, True, False)
    plt.show()
    return l_pos