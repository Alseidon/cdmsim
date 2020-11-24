# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt

from time_handler import simulator

N = 2
L = 10
m = 10

t = np.logspace(-2, 3, 50)
a = np.sqrt(t)

pos_init = random((N, 3)) * L
mom_init = (random((N, 3)) - 0.5) * 5
mom_init = np.zeros((N,3))

simulator(L, m, a, pos_init, mom_init)