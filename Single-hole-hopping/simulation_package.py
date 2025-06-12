############################################################################# Packages

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import matplotlib.colors
import seaborn as sns
import kwant
import kwant.continuum
import scipy
from scipy.constants import physical_constants
import time
import json
import copy
import pandas as pd


############################################################################# Plot settings

sns.set_style('white')
cmap_potential = 'vlag'
cmap_wavefunction = 'icefire_r'

plt.rcParams.update({'figure.figsize': (15, 6),
                     'font.family': 'serif',
                     'axes.linewidth': 0.8,
                     'legend.frameon': False,
                     'legend.loc': 'best',
                     'savefig.bbox': 'tight',
                     'legend.title_fontsize': 16,
                     'xtick.labelsize': 20,
                     'ytick.labelsize': 20,
                     'axes.labelsize': 20,
                     'axes.titlesize': 20,
                     'figure.titlesize': 20,
                     'legend.fontsize': 14,
                     'lines.marker': 'x',
                     'lines.markersize': 10
                    })

############################################################################# Constants

EH = 1e3 * physical_constants["Hartree energy in eV"][0]    # Hartree energy [meV]
a0 = 1e9 * physical_constants["Bohr radius"][0]             # Bohr radius [nm]
muB = 1e3 * physical_constants["Bohr magneton in eV/T"][0]  # Bohr magneton [meV/T]
phi0 = 2e18 * physical_constants["mag. flux quantum"][0]    # magnetic flux h/e [T*nm^2]

############################################################################# Matrix definitions - Provided by Vincent Philippe Michal

id4 = np.identity(4)
sqrt3 = np.sqrt(3)

# Define spin-3/2 matrices
Jx = np.array([[0, sqrt3/2, 0, 0], 
               [sqrt3/2, 0, 1, 0], 
               [0, 1, 0, sqrt3/2], 
               [0, 0, sqrt3/2, 0]])

Jy = np.array([[0, -1j*sqrt3/2, 0, 0], 
               [1j*sqrt3/2, 0, -1j, 0], 
               [0, 1j, 0, -1j*sqrt3/2], 
               [0, 0, 1j*sqrt3/2, 0]])

Jz = np.array([[3/2, 0, 0, 0], 
               [0, 1/2, 0, 0], 
               [0, 0, -1/2, 0], 
               [0, 0, 0, -3/2]])

Jyz = np.matmul(Jy, Jz) + np.matmul(Jz, Jy)
Jzx = np.matmul(Jz, Jx) + np.matmul(Jx, Jz)
Jxy = np.matmul(Jx, Jy) + np.matmul(Jy, Jx)
Jx2 = np.matmul(Jx, Jx)
Jy2 = np.matmul(Jy, Jy)
Jz2 = np.matmul(Jz, Jz)
Jx3 = np.matmul(Jx2, Jx)
Jy3 = np.matmul(Jy2, Jy)
Jz3 = np.matmul(Jz2, Jz)

############################################################################# Material parameters - Provided by Vincent Philippe Michal

#Luttinger and strain parameters gamma1 gamma2 gamma3 kappa qu bv[meV] nu=2*c_12/c_11 Ck[meV.nm] C4[meV.nm] / materials Si Ge InP GaAs InAs InSb
parameters = [{"Material": "Si", "ga1": 4.285, "ga2": 0.339, "ga3": 1.446, "ka": -0.42, "qu": 0.01, "bv": -2.1e3, "nu": 0.77},
              {"Material": "Ge", "ga1": 13.38, "ga2": 4.24, "ga3": 5.69, "ka": 3.41, "qu": 0.06, "bv": -2.86e3, "nu": 0.73},
              {"Material": "InP", "ga1": 4.95, "ga2": 1.65, "ga3": 2.35, "ka": 0.97, "qu": 0},
              {"Material": "GaAs", "ga1": 6.85, "ga2": 2.1, "ga3": 2.9, "ka": 1.2, "qu": 0.01},
              {"Material": "InAs", "ga1": 20.4, "ga2": 8.3, "ga3": 9.1, "ka": 7.6, "qu": 0.04, "bv":-1.8e3, "Ck": -1.12, "C4": 7e2},
              {"Material": "InSb", "ga1": 37.1, "ga2": 16.5, "ga3": 17.7, "ka": 15.6, "qu": 0.39}]

def get_material_parameters(mat):
    for row in parameters:
        if row["Material"] == mat:
            return row
        
Eg = (0.742 * 1e3) # Germanium band gap [meV]

        
############################################################################# Vector potential definition

def potential_vector_x(y, z, args):
    return (2*np.pi/phi0) * (z * args['B'][1] - y * args['B'][2])

def potential_vector_y(z, args):
    return (2*np.pi/phi0) * (- z * args['B'][0])

############################################################################# Hamiltonian - Provided by Vincent Philippe Michal

mu = 0.5 * EH * a0**2

H_LK = (
    " mu * ((ga1 + 5*ga2/2) * (k_x**2 + k_y**2 + k_z**2 + 2 * k_x * Ax(y,z,args) + 2 * k_y * Ay(z,args)) * id4 "
    " - 2 * ga2 * (k_x**2 * Jx2 + k_y**2 * Jy2 + k_z**2 * Jz2 + 2 * k_x * Ax(y,z,args) * Jx2 + 2 * k_y * Ay(z,args) * Jy2) "
    " - ga3 * (kyz * Jyz + kzx * Jzx + kxy * Jxy "
    " + (k_z * Ay(z,args) + Ay(z,args) * k_z) * Jyz + (k_z * Ax(y,z,args) + Ax(y,z,args) * k_z) * Jzx "
    " + (k_x * Ay(z,args) + Ax(y,z,args) * k_y + k_y * Ax(y,z,args) + Ay(z,args) * k_x) * Jxy)) "
    " + 2 * muB * (ka * (Jx*Bx + Jy*By + Jz*Bz) + qu * (Jx3*Bx + Jy3*By + Jz3*Bz)) "
    " + bv * ( epsxx * Jx2 + epsyy * Jy2 + epszz * Jz2) "
    " + V(x, y, z, args) * id4") 

############################################################################# Gaussian potential

def gaussian_potential(x0, y0, V, s):
    '''
    Calculate the gaussian potential at a given point (x0, y0).

    Parameters:
        x0, y0: float
            The coordinates of the point at which to calculate the potential.
        args: dict
            Dictionary containing the parameters of the potential. Must contain:
            V: float
                The height of the potential.
            s: float
                The width of the potential.
    
    Returns:
            The potential at the point (x0, y0).
    '''
    return  V * np.exp(-0.5 * (x0**2 + y0**2) / s**2)


def double_dot_gaussian_potential(x, y, z, args):
    '''
    Calculate the double dot potential at a given point (x,y,z) in the double dot system.
    Parameters:
        x, y, z: float
            The coordinates of the point at which to calculate the potential.
        args: dict
            Dictionary containing the parameters of the potential. Must contain:
            Lx, Ly: float
                The size of the system in the x and y directions.
            V1, s1 (V2, s2:): float
                The height and width of the Gaussian potentials.
            E: tuple of floats
                Electric field in x, y, and z directions.
            d: float
                The distance between the centers of the two dots.
    Returns:
            The potential at the point (x, y, z).
    '''
    z = z + args['zdist']
    xleft_center = (args['Lx'] - args['d'])/2 - 1   # [length, but indexed from zero]
    xright_center = (args['Lx'] + args['d'])/2 - 1  # [length, but indexed from zero]
    y_center = args['Ly']/2 - 1                     # [length]
    E = args['E']

    return  gaussian_potential(x-xleft_center,y-y_center, args['V1'], args['s1']) \
                + gaussian_potential(x-xright_center, y-y_center, args['V2'], args['s2']) + E[0]*x + E[1]*y + E[2]*z 


############################################################################# Potential derived analytically from J. Appl. Phys. 77, 4504–4512 (1995)

def analytical_potential(x, y, z, x0, y0, V, sx, sy, args):
    '''
    Calculate the potential at a given point (x,y,z) in the double dot system.

    Parameters:
        x, y, z: float
            The coordinates of the point at which to calculate the potential.
        x0, y0: float
            The center of the dot.
        V: float
            Potential strength.
        sx, sy: float
            Width of the dot in x and y direction.
        zdist: float
            Distance from gates to regime in question.
    Returns:
            The potential at the point (x, y, z). 
    '''
    z = z + args['zdist']
    
    def g(x,y,z):
        result = 0.5 / np.pi * np.arctan2(x * y, (z * np.sqrt(x**2 + y**2 + z**2)))
        return result 
    
    L = x0 - sx/2 # Left and right x-coordinates of the dot
    R = x0 + sx/2  
    B = y0 - sy/2 # Bottom and top y-coordinates of the dot
    T = y0 + sy/2  

    V_total = g(x-L, y-B, z) + g(x-L, T-y, z) + g(R-x, y-B, z) + g(R-x, T-y, z)

    return V * V_total


def single_dot_analytical_potential(x, y, z, args):
    '''
    Calculate the potential at a given point (x,y,z) in the single dot system, including the effect from a barrier gate.
    Parameters:
        x, y, z: float
            The coordinates of the point to calculate the potential.
        args: dict
            Dictionary containing the parameters of the potential. Must contain:
            Lx, Ly: float
                The size of the full system in the x and y directions.
            d: float
                The distance between the centers of the two dots.
            V1, s1: float
                Potential strength and width of the Gaussian potential centered at (Lx/2, Ly/2).
            Vb: float
                Potential strength of the barrier gate potential.
    Returns:
            The potential at the point (x, y, z).
    '''

    # Define the center of the 
    x_left_center = args['Lx']/2 - 1
    y_center = args['Ly']/2 - 1 
    x_barrier_gate_center = args['Lx'] - 1

    V_total = (analytical_potential(x, y, z, x_left_center, y_center, args['V1'], args['s1x'], args['sy'], args)
               + analytical_potential(x, y, z, x_barrier_gate_center, y_center, args['Vb'], args['sb'], args['sy'], args))
    
    return V_total 


# Combine it to a double dot potential
def double_dot_analytical_potential(x, y, z, args):
    '''
    Calculate the potential at a given point (x,y,z) in the double dot system, including the effect from a barrier gate.

    Parameters:
        x, y, z: float
            The coordinates of the point at which to calculate the potential.
        args: dict
            Lx, Ly: float
                The size of the system in the x and y directions.
            d: float
                The distance between the centers of the two dots.
            V1, s1 (V2, s2): float
                The strength of the effective potentials, and the diameters of the gates.
            Vb: float
                The height of the barrier gate potential.
    Returns:
            The potential at the point (x, y, z).
    '''

    # Define the centers of the two dots
    x_left_center, x_right_center = (args['Lx']-args['d'])/2 - 1, (args['Lx']+args['d'])/2 - 1
    y_center = args['Ly']/2  - 1 
    x_barrier_gate_center = args['Lx']/2  - 1

    if args['s1x']/2 + args['s2x']/2 > args['d']:
        raise ValueError('s1x/2+s2x/2 > d, so the gates overlap physically in the x-direction')

    V_total = (analytical_potential(x, y, z, x_left_center, y_center, args['V1'], args['s1x'], args['sy'], args)
               + analytical_potential(x, y, z, x_right_center, y_center, args['V2'], args['s2x'], args['sy'], args)
               + analytical_potential(x, y, z, x_barrier_gate_center, y_center, args['Vb'], args['sb'], args['sy'], args))
    
    return V_total


############################################################################# Visualize potential 

def visualize_potential_2D(potential_function, simulation_params):
    '''
    Visualize the  given potential function in 2D for given simulation parameters.
    Parameters:
        potential_function: function
            The potential function to visualize.
        simulation_params: dict
            A dictionary containing the parameters of the simulation.
    Returns:
        fig, ax:
            The 2D figure of the potential.
    '''

    Lx, Ly, Lz = simulation_params['Lx'], simulation_params['Ly'], simulation_params['Lz']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']
    
    # Make 3D meshgrid to calculate the potential on
    x = np.arange(0, Lx, ax)
    y = np.arange(0, Ly, ay)
    z = np.arange(0, Lz, az)
    X, Y, Z = np.meshgrid(x, y, z) 
    
    V_plot = potential_function(X, Y, Z, simulation_params)
    
    # Create shared colorbar 
    vmin, vmax = np.min(V_plot), np.max(V_plot)
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)  
    sm = ScalarMappable(norm = norm, cmap = cmap_potential)
    sm.set_array([])  # dummy array, required

    fig, axs = plt.subplots(nrows = 1, ncols = 3, gridspec_kw={'wspace': 0.3})
    axs = axs.flatten()

    for i, z in enumerate(np.linspace(0, Lz/az-2, 3, dtype = int)):
        axs[i].set(xlabel = r"$x$ [nm]", ylabel = r"$y$ [nm]", title = f'z = {z*az}nm')
        axs[i].set_aspect('equal')
        image =  V_plot[:, :, z] 
        p = axs[i].contourf(image, levels = 50, cmap = cmap_potential, norm = norm)
        CS = axs[i].contour(image, levels = 5, colors = 'k', linewidths = 0.8)
        axs[i].clabel(CS, fmt = '%.1f', fontsize = 8)

        axs[i].set_xticks(np.arange(0, Lx/ax, 10))
        axs[i].set_xticklabels(np.arange(0, Lx, 10*ax))
        axs[i].set_yticks(np.arange(0, Ly/ay, 10))
        axs[i].set_yticklabels(np.arange(0, Ly, 10*ay))
        axs[i].tick_params(axis = 'x', labelrotation = 45)

    cbar = fig.colorbar(sm, ax = axs, orientation = 'horizontal', fraction = 0.05, pad = 0.25)
    cbar.set_label(r"Potential [meV]")
    cbar.ax.tick_params(rotation = 45)

    return fig, axs



def visualize_potential_3D(potential_function, simulation_params):
    '''
    Visualize the given potential function in 3D for given simulation parameters.

    Parameters:
        potential_function: function
            The potential function to visualize.
        simulation_params: dict
    
    Returns:
        fig, ax:
            The 3D figure of the potential.

    '''

    Lx, Ly, Lz = simulation_params['Lx'], simulation_params['Ly'], simulation_params['Lz']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']
    
    # 3D meshgrid
    x = np.arange(0, Lx, ax)
    y = np.arange(0, Ly, ay)
    z = np.arange(0, Lz, az)
    X, Y, Z = np.meshgrid(x, y, z) 

    V_plot = potential_function(X, Y, Z, simulation_params)

    # Compute global min and max values across all z slices
    vmin, vmax = np.min(V_plot), np.max(V_plot)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # Shared normalization

    fig_3D = plt.figure() 
    axes_list = [] 

    for i,z in enumerate(np.linspace(0, Lz/az-2, 3, dtype = int)):
        axes = fig_3D.add_subplot(1, 3, i+1, projection='3d')
        axes_list.append(axes) 
        surface = axes.plot_surface(X[:,:,z], Y[:,:,z], V_plot[:,:,z],
                                    cmap = cmap_potential, norm = norm, edgecolor='none')
        axes.set(title = f'z={z*az}nm')
        axes.set_xlabel('x [nm]', labelpad = 8)
        axes.set_ylabel('y [nm]', labelpad = 8)

    axes.set_zlim(vmin, vmax)

    cbar = fig_3D.colorbar(surface, ax=axes_list, orientation='horizontal', fraction=0.05, pad=0.15)
    cbar.set_label('Potential [meV]')

    return fig_3D, axes



def visualize_potential_xy(potential_function, simulation_params, distance_from_qw_top = 0):
    '''
    Visualize the potential in the xy-plane at a given z-value.

    Parameters:
        potential_function: function
            The potential function to visualize.
        simulation_params: dict
            A dictionary containing the parameters of the simulation.
        distance_from_qw_top: int
            Distance from the top of the quantum well to the plane in which to visualize the potential.
    
    Returns:
        fig, ax:
            The 3D figure of the potential.
    '''

    simulation_params_plot = copy.deepcopy(simulation_params)
    simulation_params_plot['zdist'] = 0
    Lx, Ly = simulation_params_plot['Lx'], simulation_params_plot['Ly']
    x, y, z = np.arange(0, Lx, 1), np.arange(0, Ly, 1), np.arange(0, 100, 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_plot = potential_function(X, Y, Z, simulation_params_plot)

    fig, ax = plt.subplots()
    p = ax.contourf(V_plot[:,:,distance_from_qw_top+simulation_params['zdist']].T*1e-3,
                    levels = 50, cmap = cmap_potential)
    CS = ax.contour(V_plot[:,:,distance_from_qw_top+simulation_params['zdist']].T*1e-3,
                    levels = 8, colors = 'k', linewidths = 0.8)
    
    ax.clabel(CS, fmt = '%.1f', fontsize = 14)
    cbar = fig.colorbar(p, orientation='vertical')
    cbar.set_label(r"Potential [V]")
    ax.set(xlabel = r"$x$ [nm]", ylabel = r"$y$ [nm]")

    cbar.ax.tick_params(rotation=45)
    fig.suptitle('z = ' + str(distance_from_qw_top) + 'nm')
    fig.tight_layout()

    return fig, ax


def visualize_potential_xz(potential_function, simulation_params):
    '''
    Visualize the potential in the xz-plane at a given z-value.

    Parameters:
        potential_function: function
            The potential function to visualize.
        simulation_params: dict
            A dictionary containing the parameters of the simulation.
    
    Returns:
        fig, ax:
            The 3D figure of the potential.
    '''
        
    simulation_params_plot = copy.deepcopy(simulation_params)
    simulation_params_plot['zdist'] = 0

    Lx, Ly = simulation_params_plot['Lx'], simulation_params_plot['Ly']
    x, y, z = np.arange(0, Lx, 1), np.arange(0, Ly, 1), np.arange(0, 100, 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_plot = potential_function(X, Y, Z, simulation_params_plot)

    fig, ax = plt.subplots()
    im = V_plot[:,int(Ly/2),:].T*1e-3
    plot = ax.pcolor(im, cmap = cmap_potential) 
    ax.invert_yaxis()
    
    cbar = fig.colorbar(plot, ax=ax)
    cbar.set_label('Potential [V]', fontsize=20)
    
    # Plot quantum well
    ax.axhspan(simulation_params['zdist'], simulation_params['zdist']+simulation_params['Lz'],
               facecolor='none', hatch='///', edgecolor='black', alpha = 0.6, linewidth=0.0, label = 'Quantum well') 
    ax.set(xlabel = 'x [nm]', ylabel = 'z [nm]')
    ax.legend(prop = {'size': 20}, frameon = True, facecolor = 'white', loc = 'center right')
    fig.suptitle(f'y = {int(Ly/2)}nm')
    fig.tight_layout()

    return fig, ax




############################################################################# System definition

def get_syst(ham, params, mat = 'Ge'):
    '''
    Construct the system for the given Hamiltonian and parameters in the given material.
    The function assumes the system is a rectangular box with the shape Lx x Ly x Lz.

    Parameters:
        ham: str
            The Hamiltonian to use for the system.
        params: dict
            A dictionary containing the parameters of the simulation.
        mat: str
            The material to use for the system. Default is 'Ge'.

    Returns:
        syst: kwant.builder.FiniteSystem
            The finalized system. 
    '''
    
    def box_3D(site):
        (x, y, z) = site.pos
        return 0 <= x < params['Lx'] and 0 <= y < params['Ly'] and 0 <= z < params['Lz']

    # Define all the "constants" and substitutions in the Hamiltonian
    mat_params = get_material_parameters(mat)
    consts = { "mu": mu,
              "muB": muB,
              "phi0": phi0}

    subs = { "Bx": params['B'][0], "By": params['B'][1], "Bz": params['B'][2],
                  "kyz": "k_y * k_z + k_z * k_y",
                  "kzx": "k_z * k_x + k_x * k_z",
                  "kxy": "k_x * k_y + k_y * k_x",
                  "id4": id4, "Jx": Jx, "Jy": Jy, "Jz": Jz,
                  "Jx2": Jx2, "Jy2": Jy2, "Jz2": Jz2,
                  "Jyz": Jyz, "Jzx": Jzx, "Jxy": Jxy,
                  "Jx3": Jx3, "Jy3": Jy3, "Jz3": Jz3,
                  'epsyy': params['epsxx'],
                  'epszz': (-mat_params['nu'] * params['epsxx']),
                  **mat_params, **consts}

    # Actually build the system 
    syst = kwant.Builder()
    model = kwant.continuum.discretize(ham, locals=subs, coords = ('x', 'y', 'z'),
                                       grid=kwant.lattice.general([(params['ax'], 0, 0),
                                                                   (0, params['ay'], 0),
                                                                   (0, 0, params['az'])],
                                                                    norbs = 4)) # definition of primitive vectors
    
    syst.fill(model, box_3D, (0, 0, 0))
    syst = syst.finalized()
    
    return syst


############################################################################# Solve system 


def get_eigenvalues_and_eigenvectors(potential_function, simulation_params, mat = 'Ge', k = 30, return_eigenvectors = True):
    '''
    Calculate the eigenvalues and eigenvectors of the Hamiltonian for the given potential function and simulation parameters.

    Parameters:
        potential_function: function
            The potential function to use for the simulation.
        simulation_params: dict
            A dictionary containing the parameters of the simulation.
        mat: str
            The material to use for the system. Default is 'Ge'.
        k: int
            The number of eigenvalues to calculate. Default is 30.
        return_eigenvectors: bool
            Whether to return the eigenvectors or not. Default is True.
        
    Returns:
        eigenvalues: np.ndarray
            The eigenvalues of the Hamiltonian.
        eigenvectors: np.ndarray
            The eigenvectors of the Hamiltonian. Only returned if return_eigenvectors is True, otherwise None.
    '''

    ham = H_LK
    syst = get_syst(ham, simulation_params, mat)
    sim_params = simulation_params.copy()
    sim_params.update(V=potential_function, Ax=potential_vector_x, Ay=potential_vector_y, args = simulation_params)
    
    t1 = time.time()
    ham = syst.hamiltonian_submatrix(params=sim_params, sparse=True)
    t2 = time.time()
    print(f'Hamiltonian shape: {ham.shape}. Construction time: {round((t2 - t1) ,2)} s.')

    if return_eigenvectors:
        t1 = time.time()
        evals, evecs = scipy.sparse.linalg.eigsh(ham, k=k, which='SA') 
        t2 = time.time()
        print('Solve time:', round((t2 - t1) ,2), 's.')

        idx = np.argsort(evals)      # sort the eigenvalues
        eigenvalues = evals[idx]     # sort the eigenvalues
        eigenvectors = evecs[:,idx]  # sorting the eigenvectors

        print('Four lowest eigenvalues:', eigenvalues[:4])
        return eigenvalues, eigenvectors
    
    else:
        t1 = time.time()
        evals = scipy.sparse.linalg.eigsh(ham, k=k, which='SA', return_eigenvectors = False) 
        t2 = time.time()
        print('Solve time:', round((t2 - t1) ,2), 's.')

        idx = np.argsort(evals)      # sort the eigenvalues
        eigenvalues = evals[idx]     # sort the eigenvalues

        print('Four lowest eigenvalues:', eigenvalues[:4])

        return eigenvalues, None



############################################################################# Visualize eigenstates



def visualize_total_wavefunction_xy(simulation_params, eigenstates, n_state):
    '''
    Visualize the total wavefunction at varying z-values in the xy-plane.
    Parameters:
        simulation_params: dict
            A dictionary with the parameters of the simulation.
        eigenstates: np.ndarray
            The eigenvectors of the Hamiltonian, achieved from the solve function.
        n_state: int
            The index of the state to visualize.

    Returns:
        fig, ax:
            The figure of the total wavefunction.
    '''

    norbs = 4                              # Degrees of freedom
    n_state_evec = eigenstates[:, n_state] # Extracting the n-th state eigenvector

    # Reshape the eigenvector because there are multiple orbitals per site
    n_state_wavefunction = n_state_evec.reshape(int(len(n_state_evec)/norbs), norbs)
    
    Lx, Ly, Lz = simulation_params['Lx'], simulation_params['Ly'], simulation_params['Lz']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']

    norm = np.sum(np.conj(n_state_wavefunction) * n_state_wavefunction) * ax*ay*az
    n_state_wavefunction /= np.sqrt(norm)

    abs_psi_squared = 0
    for i in range(4):
        abs_psi_squared += np.abs(n_state_wavefunction[:,i].reshape(int(Lx/ax), int(Ly/ay), int(Lz/az)))**2

    fig, axes = plt.subplots(1, 3, sharey = True)
    axes = axes.flatten()

    for i,z in enumerate([0,2,4]):
        image = abs_psi_squared[:, :, z].T
        image /= np.max(image)
        p = axes[i].contourf(image, levels = 30, cmap = cmap_wavefunction)
        axes[i].contour(image, levels = 5, colors = 'k', linewidths = 0.8)
        axes[i].set(xlabel = r"$x$ [nm]", title = f'z = {z*az}nm')
        axes[i].set_aspect('equal')
        cb = fig.colorbar(p, ax = axes[i], orientation="horizontal", pad= 0.3, shrink=0.9)
        cb.ax.tick_params(rotation=45)
        cb.set_label(r"$|\psi(\mathbf{r})|^2 / |\psi_{\max}|^2$")

        if i == 0:
            axes[i].set( ylabel = r"$y$ [nm]")

    
        # Change xticks and yticks by mulitplying by the lattice constant, so they are physically correct
        axes[i].set_xticks(np.arange(0, Lx/ax, 10))
        axes[i].set_xticklabels(np.arange(0, Lx, 10*ax))
        axes[i].set_yticks(np.arange(0, Ly/ay, 10))
        axes[i].set_yticklabels(np.arange(0, Ly, 10*ay))
        axes[i].tick_params(axis = 'x', labelrotation = 45)

        cb.ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    fig.suptitle(r'$\psi_%s$' % n_state)
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()

    return fig, axes



def visualize_total_wavefunction_yz(simulation_params, eigenstates, n_state):
    '''
    Visualize the total wavefunction at varying z-values in the yz-plane.
    Parameters:
        simulation_params: dict
            A dictionary with the parameters of the simulation.
        eigenstates: np.ndarray
            The eigenvectors of the Hamiltonian, achieved from the solve function.
        n_state: int
            The index of the state to visualize.

    Returns:
        fig, ax:
            The figure of the total wavefunction.
    '''

    norbs = 4  
    n_state_evec = eigenstates[:, n_state] 
    n_state_wavefunction = n_state_evec.reshape(int(len(n_state_evec)/norbs), norbs)
   

    Lx, Ly, Lz = simulation_params['Lx'], simulation_params['Ly'], simulation_params['Lz']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']

    norm = np.sum(np.conj(n_state_wavefunction) * n_state_wavefunction) * ax*ay*az
    n_state_wavefunction /= np.sqrt(norm)

    gate_1_mid = int(Lx/2 - simulation_params['d']/2) / ax
    gate_2_mid = int(Lx/2 + simulation_params['d']/2) / ax
    
    abs_psi_squared = 0
    for i in range(4):
        abs_psi_squared += np.abs(n_state_wavefunction[:,i].reshape(int(Lx/ax), int(Ly/ay), int(Lz/az)))**2

    fig, axes = plt.subplots(1, 3, sharey = True)
    axes = axes.flatten()

    for i, x in enumerate([int(gate_1_mid), int(Lx/2/ax), int(gate_2_mid)]):
        image = abs_psi_squared[x, :, :].T
        image /= np.max(image)
        p = axes[i].contourf(image, levels = 30, cmap = cmap_wavefunction)
        axes[i].contour(image, levels = 8, colors = 'k', linewidths = 0.8)
        axes[i].set(xlabel = r"$y$ [nm]", title = f'x = {x*ax}nm')
        axes[i].set_aspect('equal')
        cb = fig.colorbar(p, ax = axes[i], orientation="horizontal", pad= 0.3, shrink=0.9)
        cb.ax.tick_params(rotation=45)
        cb.set_label(r"$|\psi(\mathbf{r})|^2 / |\psi_{\max}|^2$")

        if i == 0:
            axes[i].set(ylabel = r"$z$ [nm]")
    
        # Change xticks and yticks by mulitplying by the lattice constant, so they are physically correct
        axes[i].set_xticks(np.arange(0, Ly/ay, 5))
        axes[i].set_xticklabels(np.arange(0, Ly, 5*ay))
        axes[i].set_yticks(np.arange(0, Lz/az, 2))
        axes[i].set_yticklabels(np.arange(0, Lz, 2*az))
        axes[i].tick_params(axis = 'x', labelrotation = 45)
        cb.ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    fig.suptitle(r'$\psi_%s$' % n_state)
    fig.tight_layout()


############################################################################# Computations


def compute_spin_split(eigenvalues):
    ''' Compute the spin gap between the two lowest states '''
    return eigenvalues[1] - eigenvalues[0]


def compute_charge_split(eigenvalues):
    ''' Compute the charge gap between the two lowest states '''
    return ((eigenvalues[2] + eigenvalues[3]) - (eigenvalues[0] + eigenvalues[1]) ) / 2


def compute_interdot_distance(simulation_params, potential_function):
    Lx, Ly = simulation_params['Lx'], simulation_params['Ly']
    z_top = simulation_params['zdist']
    y_mid = int(Ly/2)

    x = np.arange(0, Lx, 1)
    y = np.arange(0, Ly, 1)
    z = np.arange(0, 100, 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    analytical_params_dd_extra = copy.deepcopy(simulation_params)
    analytical_params_dd_extra['zdist'] = 0
    V = potential_function(X, Y, Z, analytical_params_dd_extra)

    V_left_min =  np.argmin(V[:int(Lx/2), y_mid, z_top])                     # Index of potential minimum on the left side
    V_right_min = np.argmin((V[int(Lx/2):, y_mid, z_top])) + Lx/2            # Index of potential minimum on the right side
    x_LR = V_right_min - V_left_min              # Distance between the two minima in nm

    return x_LR


def compute_dipole_transition_matrix_element(simulation_params, eigenstates, dimension = 'x', norbs = 4):

    Lx, Ly, Lz = simulation_params['Lx'], simulation_params['Ly'], simulation_params['Lz']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']
    norbs = 4

    x = np.arange(0, Lx, ax)
    y = np.arange(0, Ly, ay)
    z = np.arange(0, Lz, az)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid_shape = np.shape(X)
    psi0 = eigenstates[:, 0].reshape((int(Lx/ax) * int(Ly/ay) * int(Lz/az), norbs))
    psi1 = eigenstates[:, 1].reshape((int(Lx/ax) * int(Ly/ay) * int(Lz/az), norbs))

    # Normalize
    norm2_psi0 = np.sum(np.conj(psi0) * psi0) * ax*ay*az
    psi0_normalized = psi0 / np.sqrt(norm2_psi0)
    
    norm2_psi1 = np.sum(np.conj(psi1) * psi1) * ax*ay*az
    psi1_normalized = psi1 / np.sqrt(norm2_psi1)

    if dimension == 'x':
        R = X
    elif dimension == 'y':
        R = Y
    elif dimension == 'z':
        R = Z
        
    # Compute matrix element for each spinor component
    integral = 0.0 + 0.0j 
    for i in range(4):
        psi0_i = psi0_normalized[:, i].reshape(grid_shape)
        psi1_i = psi1_normalized[:, i].reshape(grid_shape)

        integrand = np.conj(psi0_i) * R * psi1_i
        integral += np.sum(integrand) * ax * ay * az 

    x01 = np.abs(integral)

    return x01


def compute_metrics_of_interest(simulation_params, eigenvalues, eigenstates): 
    delta_s = compute_spin_split(eigenvalues)
    delta_c = compute_charge_split(eigenvalues)

    # Calculate dipole transition matrix element
    x01 = compute_dipole_transition_matrix_element(simulation_params = simulation_params,
                                             eigenstates = eigenstates)

    interdot_distance = compute_interdot_distance(simulation_params = simulation_params,
                                                 potential_function = double_dot_analytical_potential)

    d01 = x01 * 2 / interdot_distance
    tau = d01 * delta_c / delta_s
    eta = np.arctan(tau) * 2 / np.pi

    df = pd.DataFrame({"Parameter": ["delta_s", "delta_c", "interdot_distance", "x_01", "d01", "tau", "eta"],
                   "Value": [delta_s*1e3, delta_c*1e3, interdot_distance, x01, d01, tau, eta],
                   "Unit": ["ueV", "ueV", "nm", "nm", "", "", "pi/2"]})
    
    return df


def plot_device_layout(simulation_params):

    Lx, Ly = simulation_params["Lx"], simulation_params["Ly"]
    s1x, s2x = simulation_params["s1x"], simulation_params["s2x"]
    sb = simulation_params["sb"]
    sy = simulation_params["sy"]
    d = simulation_params["d"]

    #  Center x positions
    left_gate_center_x = (Lx - d) / 2
    right_gate_center_x = left_gate_center_x + d
    barrier_gate_center_x = (left_gate_center_x + right_gate_center_x) / 2

    # Bottom-left x positions
    left_gate_x = left_gate_center_x - s1x / 2
    right_gate_x = right_gate_center_x - s2x / 2
    barrier_gate_x = barrier_gate_center_x - sb / 2

    # Center y position
    y_mid = Ly / 2 - sy / 2   

    fig, ax = plt.subplots()#frameon=False)

    # Full computational domain
    computational_domain = patches.Rectangle((0, 0), Lx, Ly, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(computational_domain)
    ax.text(Lx/2, 25, r'$L_x$=' + f'{Lx}nm', ha = 'center', va = 'top', fontsize = 20)
    ax.text(25, Ly/2, r'$L_y$=' + f'{Ly}nm', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)

    # Left gate 
    left_gate = patches.Rectangle((left_gate_x, y_mid), s1x, sy, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(left_gate)
    ax.text(left_gate_center_x, y_mid + sy + 1, r'$s_{1x}$=' + f'{s1x}nm', ha='center', va='bottom', fontsize = 14)

    # Barrier gate
    barrier_gate = patches.Rectangle((barrier_gate_x, y_mid), sb, sy, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(barrier_gate)
    ax.text(barrier_gate_center_x, y_mid + sy + 1, r'$s_{b}$=' + f'{sb}nm', ha='center', va='bottom', fontsize = 14)

    # Right gate
    right_gate = patches.Rectangle((right_gate_x, y_mid), s2x, sy, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(right_gate)
    ax.text(right_gate_center_x, y_mid + sy + 1, r'$s_{2x}$=' + f'{s2x}nm', ha='center', va='bottom', fontsize = 14)

    ax.text(Lx/2-s1x-sb/2, Ly/2, r'$s_{y}$=' + f'{sy}nm', ha='right', va='center', rotation='vertical', fontsize = 14)
    ax.annotate('', xy = (left_gate_center_x, Ly/2), xytext = (right_gate_center_x, Ly/2),
                arrowprops = dict(arrowstyle = '<->', color = 'k'))
    ax.text((left_gate_center_x + right_gate_center_x) / 2, Ly/2-2, f'd={d}nm', ha='center', va='top', color='k', fontsize=14)

    ax.set(xlim = (-1, Lx+1), ylim = (-1, Ly+1))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax