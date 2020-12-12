# -*- coding: utf-8 -*-
import numpy as np
import numpy.fft as fft

omega0 = 1

def G(kx, ky, kz, a, om=omega0):
    """
    Green function for the Poisson equation

    Parameters
    ----
    - kx, ky, kz : L arrays
        spatial pulsations in every cartesian direction

    - a : real
        scale factor

    - om : real, optional
        value of omega0. default : om0

    Returns
    -------
    - mat : L*L*L matrix
        G evaluated at each 3D point in k-space, 0 at origin
    """
    n = len(kx)
    mat = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if kx[i]==0 and ky[j]==0 and kz[k]==0:
                    mat[i,j,k] = 0
                else:
                    mat[i,j,k] = -3*om / (8*a * (np.sin(kx[i] / 2)**2 + np.sin(ky[j] / 2)**2 + np.sin(kz[k] / 2)**2))
    return mat


def psolve(delta, a, om=omega0):
    """
    Solves Nabla phi = 3 om / (2a) * delta
    
    Parameters
    ----
    - delta : L*L*L array
        density

    - a : real
        scale factor

    - om : real, optional
        value of omega0. default : om0
    
    Returns
    -------
    - 0 #TO UPDATE

    Notes
    -----
    Goes in Fourier space. Will modify the delta array to input the solution phi
    """
    delta = fft.fftn(delta)
    L = np.shape(delta)[0]
    kx, ky, kz = fft.fftfreq(L), fft.fftfreq(L), fft.fftfreq(L)
    delta *= G(kx, ky, kz, a, om)
    delta = fft.ifftn(delta).real
    return delta