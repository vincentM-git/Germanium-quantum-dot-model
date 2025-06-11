#!/usr/bin/env python

import numpy as np
import kwant
import scipy.sparse.linalg as sla
from single_dot import Gaussian_potential, single_dot_potential


def Gaussian_symmetric_double_dot(x, y, d, V0, s):
    return Gaussian_potential(x - d/2, y, V0, s) + Gaussian_potential(x + d/2, y, V0, s)
    

def double_dot_potential(site, x1, y1, V1, s1, x2, y2, V2, s2):
    return single_dot_potential(site, x1, y1, V1, s1) + single_dot_potential(site, x2, y2, V2, s2)


def make_double_dot(W, L, a=1.0, t=1.0):
    
    lat = kwant.lattice.square(a, norbs=1)
    
    syst = kwant.Builder()
    
    def onsite(site, x1, y1, V1, s1, x2, y2, V2, s2):
        return 4 * t + double_dot_potential(site, x1, y1, V1, s1, x2, y2, V2, s2)
    
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    
    syst[lat.neighbors()] = -t
    
    return syst


def get_spectrum_double_dot(syst, W, L, d, V0, s, k):
    """
    Computes the kth smallest eigenvalues of the sparse Hamiltonian matrix of the double dot system.
    """
    ham_mat = syst.hamiltonian_submatrix(params=dict(x1=(L-d)/2, y1=W/2, V1=V0, s1=s, x2=(L+d)/2, y2=W/2, V2=V0, s2=s), sparse=True)
    
    evals = sla.eigsh(ham_mat.tocsc(), k, return_eigenvectors=False, sigma=0)
    
    idx = np.argsort(evals)
    evals_sorted = evals[idx]
    
    return evals_sorted


def get_E0_E1_double_dot(syst, W, L, d, V0, s, k):
    """
    Computes the two lowest energy levels of the symmetric double quantum dot. 
    """
    evs = get_spectrum_double_dot(syst, W, L, d, V0, s, k)
    return evs[0], evs[1]


def plot_particle_density(syst, W, L, d, V0, s, n, k):
    
    ham_mat = syst.hamiltonian_submatrix(params=dict(x1=(L-d)/2, y1=W/2, V1=V0, s1=s, x2=(L+d)/2, y2=W/2, V2=V0, s2=s), sparse=True)
    evals, evecs = sla.eigsh(ham_mat.tocsc(), k, sigma=0)
    idx = np.argsort(evals)   
    #evals_sorted = evals[idx]
    evecs_sorted = evecs[:,idx]
    
    kwant.plotter.map(syst, np.abs(evecs_sorted[:, n])**2,
                      colorbar=True, oversampling=1)

