# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt

from time_handler import simulator

N = 2
L = 20
m = 10

t = np.logspace(-2, 0, 100)
a = np.sqrt(t)

pos_init = random((N, 3)) * L
mom_init = (random((N, 3)) - 0.5) * 5
mom_init = np.zeros((N,3))

real_sim = True

if real_sim:
    _ = simulator(L, m, a, pos_init, mom_init)
else:
    x = np.linspace(0, 3, 100)
    k = 2*np.pi
    w = 2*np.pi
    dt = 0.01
    t = 0
    for i in range(100):
        y = np.cos(k*x - w*t)
        if i == 0:
            line, = plt.plot(x, y)
        else:
            line.set_ydata(y)
        plt.pause(0.01) # pause avec duree en secondes
        t = t + dt
        
    plt.show()
