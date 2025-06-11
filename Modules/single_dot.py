#!/usr/bin/env python

import numpy as np
import kwant
import scipy.sparse.linalg as sla

def Gaussian_potential(x, y, V0, s):
    return  - V0 * np.exp(-0.5*(x**2 + y**2)/s**2)
     

def single_dot_potential(site, x0, y0, V0, s):
    (x, y) = site.pos
    return Gaussian_potential(x - x0, y - y0, V0, s)


def make_single_dot(L, a=1.0, t=1.0):
    
    lat = kwant.lattice.square(a, norbs=1)
    
    syst = kwant.Builder()
    
    def onsite(site, x0, y0, V0, s):
        return 4 * t + single_dot_potential(site, x0, y0, V0, s)
    
    syst[(lat(i, j) for i in range(L) for j in range(L))] = onsite
    
    syst[lat.neighbors()] = -t
    
    return syst

      
def get_spectrum_single_dot(syst, L, a, V0, s, k):
    """
    Computes the kth smallest eigenvalues of the sparse Hamiltonian matrix of the single dot system.  
    """
    
    ham_mat = syst.hamiltonian_submatrix(params=dict(x0 = a * (L - 1)/2, y0 = a * (L - 1)/2, V0 = V0, s = s), sparse=True)
    
    evals = sla.eigsh(ham_mat.tocsc(), k, return_eigenvectors=False, sigma=0)
    
    idx = np.argsort(evals)
    evals_sorted = evals[idx]
    
    return evals_sorted


def get_E0_E1_single_dot(syst, L, a, V0, s, k):
    """
    Computes the two lowest energy levels of the single quantum dot. 
    """
    evs = get_spectrum_single_dot(syst, L, a, V0, s, k)
    return evs[0], evs[1]


def plot_particle_density(syst, L, a, V0, s, n, k):
    
    ham_mat = syst.hamiltonian_submatrix(params=dict(x0 = a * (L - 1)/2, y0 = a * (L - 1)/2, V0 = V0, s = s), sparse=True)
    evals, evecs = sla.eigsh(ham_mat.tocsc(), k, sigma=0)
    idx = np.argsort(evals)   
    #evals_sorted = evals[idx]
    evecs_sorted = evecs[:,idx]
    
    kwant.plotter.map(syst, np.abs(evecs_sorted[:, n])**2,
                      colorbar=True, oversampling=1)
