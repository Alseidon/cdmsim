# -*- coding: utf-8 -*-
import numpy as np

# USING CIC INTERPOLATION
Om0, Ok0, Ol0 = 0.04, 0.2, 0.76

def f(a, om0=Om0, ok0=Ok0, ol0=Ol0):
    return np.sqrt(a / (om0 + ok0 * a + ol0 * a**3))

def mom_update(grid, pos, mom, a, da, l):
    def g(i,j,k, c, tab=grid):
        if c==0:
            return (tab[(i+1)%l,j,k] - tab[(i-1)%l,j,k])/2
        elif c==1:
            return (tab[i,(j+1)%l,k] - tab[i,(j-1)%l,k])/2
        return (tab[i,j,(k+1)%l] - tab[i,j,(k-1)%l])/2
    accs = np.zeros(mom.shape)
    for n in range(len(accs)):
        position = pos[n]
        parentcell = np.floor(position)
        i, j, k = parentcell
        i, j, k = int(i), int(j), int(k)
        t = position - parentcell
        d = 1 - t
        for c in range(3):
            accs[n, c] += (g(i,j,k,c)             * t[0]*t[1]*t[2] +
                           g((i+1)%l,j,k,c)       * d[0]*t[1]*t[2] +
                           g(i,(j+1)%l,k,c)       * t[0]*d[1]*t[2] +
                           g((i+1)%l,(j+1)%l,k,c) * d[0]*d[1]*t[2] +
                           g(i,j,(k+1)%l,c)       * t[0]*t[1]*d[2] +
                           g((i+1)%l,j,(k+1)%l,c) * d[0]*t[1]*d[2] +
                           g(i,(j+1)%l,(k+1)%l,c) * t[0]*d[1]*d[2] +
                           g((i+1)%l,(j+1)%l,(k+1)%l,c) * d[0]*d[1]*d[2] )
    mom += accs* f(a) * da
    return 0

def pos_update(pos, mom, l, a, da):
    pos += mom * (f(a + da/2) * da / (a + da/2)**2)
    pos %= l
    return 0


