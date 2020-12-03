# -*- coding: utf-8 -*-
import numpy as np

# USING CIC INTERPOLATION
Om0, Ok0, Ol0 = 0.04, 0.2, 0.76

def f(a, om0=Om0, ok0=Ok0, ol0=Ol0):
    """
    Function used in particle updates
    

    Parameters
    ----
    - a : real
        scale factor
    
    - om0, ok0, ol0 : reals, optional
        values of Omegas for matter, dark matter and cosmological constant. default : Om0, Ok0, Ol0

    Returns
    -------
    - _ : real
        result
    """
    return np.sqrt(a / (om0 + ok0 * a + ol0 * a**3))

def mom_update(grid, pos, mom, a, da, l):
    """
    Updates particles momenta

    Parameters
    ----
    - grid : L*L*L array
        value of g (minus gradient of potential) at each spatial point
    
    - pos : L*3 array
        position of each particle
    
    - mom : L*3 array
        momentum of each particle
    
    - a : real
        scale factor
    
    - da : real
        scale factor step to take
    
    - l : int
        length L of the box
    
    Returns
    ----
    - 0
    """
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

def pos_update(pos, mom, a, da, l):
    """
    Update particle positions

    Parameters
    ----
    - pos : L*3 array
        position of each particle
    
    - mom : L*3 array
        momentum of each particle
    
    - a : real
        scale factor
    
    - da : real
        scale factor step to take
    
    -l : int
        length L of the box
    
    Returns
    ----
    - 0

    """
    pos += mom * (f(a + da/2) * da / (a + da/2)**2)
    pos %= l
    return 0