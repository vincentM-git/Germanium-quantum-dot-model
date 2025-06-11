import numpy as np
from numpy import pi
import scipy.sparse.linalg as sla
import kwant
import kwant.continuum
from LK_utils import consts, Jmatrices
from double_dot import Gaussian_symmetric_double_dot
import matplotlib.pyplot as plt


def make_model_LKBP(H_LK, mat_params, lat_params, eps_para, eps_perp, B):
    syst = kwant.Builder()
    subs = consts
    subs.update(mat_params)
    subs.update(lat_params)
    subs.update(Jmatrices)
    subs.update({
        "Bx": B[0], "By": B[1], "Bz": B[2],
        "kyz": "k_y * k_z + k_z * k_y",
        "kzx": "k_z * k_x + k_x * k_z",
        "kxy": "k_x * k_y + k_y * k_x",
        "epsxx": eps_para, "epsyy": eps_para, "epszz":eps_perp
        })

    model = kwant.continuum.discretize(H_LK, locals=subs, grid=kwant.lattice.general([(lat_params['ax'], 0, 0), (0, lat_params['ay'], 0), (0, 0, lat_params['az'])], norbs=4))
    def box_3D(site):
        (x, y, z) = site.pos
        return 0 <= x < lat_params['L'] - lat_params['ax'] and 0 <= y < lat_params['W'] - lat_params['ay'] and 0 <= z < lat_params['H'] - lat_params['az']

    syst.fill(model, box_3D, (0, 0, 0))
    syst = syst.finalized()
    return syst


def eigs_gate_defined_potential(syst, x0, y0, Lg, d_QW, B, V, N=8):
    def potential_vector_x(y, z):
        return (2*pi/consts['phi0']) * (z*B[1] - y*B[2])
    
    def potential_vector_y(x, z):
        return (2*pi/consts['phi0']) * (-z*B[0])
    
    def potential(x, y, z):
        def g(x,y,z):
            return (0.5/pi) * np.arctan2(x * y, (z * np.sqrt(x**2 + y**2 + z**2)))
        
        Left = x0 - 0.5*Lg # Left and right x-coordinates of the dot
        Right = x0 + 0.5*Lg  
        Bot = y0 - 0.5*Lg # Bottom and top y-coordinates of the dot
        Top = y0 + 0.5*Lg
        return V * (g(x - Left, y - Bot, z + d_QW) + g(x - Left, Top - y, z + d_QW) + g(Right - x, y - Bot, z + d_QW) + g(Right - x, Top - y, z + d_QW))
        
    Ham = syst.hamiltonian_submatrix(params=dict(V=potential, Ax=potential_vector_x, Ay=potential_vector_y), sparse=True)
    evals, evecs = sla.eigsh(Ham, k=N, which='SA')
    idx = evals.argsort()
    return [evals[idx], evecs[:,idx]]


def eigs_double_dot_potential(syst, lat_params, B, d, s, U0, eEz, N=8):
    def potential_vector_x(y, z):
        return (2*pi/consts['phi0']) * (z*B[1] - y*B[2])
    
    def potential_vector_y(x, z):
        return (2*pi/consts['phi0']) * (-z*B[0])
    
    def double_dot_potential(x, y, z):
        return Gaussian_symmetric_double_dot(x - lat_params['L']/2, y - lat_params['W']/2, d, U0, s) + eEz*z

    Ham = syst.hamiltonian_submatrix(params=dict(V=double_dot_potential, Ax=potential_vector_x, Ay=potential_vector_y), sparse=True)
    evals, evecs = sla.eigsh(Ham, k=N, which='SA')
    idx = evals.argsort()
    return [evals[idx], evecs[:,idx]]


def D_dip(eigs, lat_params):
    L, W, H, ax, ay, az = lat_params['L'], lat_params['W'], lat_params['H'], lat_params['ax'], lat_params['ay'], lat_params['az']
    evals = eigs[0]
    Ds = evals[1] - evals[0]
    Dc = 0.5 * (evals[2] + evals[3] - evals[0] - evals[1])
    
    norbs = 4
    x = np.arange(0, L - ax, ax)
    y = np.arange(0, W - ay, ay)
    z = np.arange(0, H - az, az)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_shape = np.shape(X)
    
    # Get the wavefunctions for the first two levels
    psi0 = eigs[1][:, 0].reshape((int(L/ax - 1) * int(W/ay - 1) * int(H/az - 1), norbs))
    psi1 = eigs[1][:, 1].reshape((int(L/ax - 1) * int(W/ay - 1) * int(H/az - 1), norbs))
    
    # Compute matrix element for each spinor component
    integral_x01 = integral_y01 = integral_z01 = integral_xl = integral_yl = integral_zl = 0.0 + 0.0j 
    norm0 = norm1 = 0.0
    for i in range(4):
        psi0_i = psi0[:, i].reshape(grid_shape)
        psi1_i = psi1[:, i].reshape(grid_shape)
        norm0 += np.sum(np.conj(psi0_i)*psi0_i)
        norm1 += np.sum(np.conj(psi1_i)*psi1_i)
        integral_x01 += np.sum(np.conj(psi0_i) * X * psi1_i)
        integral_y01 += np.sum(np.conj(psi0_i) * Y * psi1_i) 
        integral_z01 += np.sum(np.conj(psi0_i) * Z * psi1_i)
        integral_xl += np.sum(np.conj(psi1_i) * X * psi1_i - np.conj(psi0_i) * X * psi0_i) 
        integral_yl += np.sum(np.conj(psi1_i) * Y * psi1_i - np.conj(psi0_i) * Y * psi0_i) 
        integral_zl += np.sum(np.conj(psi1_i) * Z * psi1_i - np.conj(psi0_i) * Z * psi0_i)
    
    evals = eigs[0]
    Dc = 0.5 * (evals[2] + evals[3] - evals[0] - evals[1])
    Ds = evals[1] - evals[0]
    
    norbs = 4
    x = np.arange(0, L - ax, ax)
    y = np.arange(0, W - ay, ay)
    z = np.arange(0, H - az, az)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_shape = np.shape(X)
    
    # Get the wavefunctions for the first two levels
    psi0 = eigs[1][:, 0].reshape((int(L/ax - 1) * int(W/ay - 1) * int(H/az - 1), norbs))
    psi1 = eigs[1][:, 1].reshape((int(L/ax - 1) * int(W/ay - 1) * int(H/az - 1), norbs))
    
    # Compute matrix element for each spinor component
    integral_x01 = integral_y01 = integral_z01 = integral_xl = integral_yl = integral_zl = 0.0 + 0.0j 
    N0 = N1 = 0.0 + 0.0j
    
    for i in range(4):
        psi0_i = psi0[:, i].reshape(grid_shape)
        psi1_i = psi1[:, i].reshape(grid_shape)
        N0 += np.sum(np.conj(psi0_i)*psi0_i)
        N1 += np.sum(np.conj(psi1_i)*psi1_i)
        integral_x01 += np.sum(np.conj(psi0_i) * X * psi1_i)
        integral_y01 += np.sum(np.conj(psi0_i) * Y * psi1_i) 
        integral_z01 += np.sum(np.conj(psi0_i) * Z * psi1_i)
        integral_xl += np.sum(np.conj(psi1_i) * X * psi1_i - np.conj(psi0_i) * X * psi0_i) 
        integral_yl += np.sum(np.conj(psi1_i) * Y * psi1_i - np.conj(psi0_i) * Y * psi0_i) 
        integral_zl += np.sum(np.conj(psi1_i) * Z * psi1_i - np.conj(psi0_i) * Z * psi0_i)

    if max(abs(N0-1), abs(N1-1))<1e-3:
        print('Normalization OK')
    
    x01 = np.abs(integral_x01)
    y01 = np.abs(integral_y01)
    z01 = np.abs(integral_z01)
    xl = np.abs(integral_xl)
    yl = np.abs(integral_yl)
    zl = np.abs(integral_zl)
    return [Dc, Ds, x01, y01, z01, xl, yl, zl]


def visualize_eigenstates(simulation_params, eigenstates, n_state, n_orbital):
    '''
    Visualize the probability density of the n-th state at different z-values, of state n_state, orbital n_orbital.

    Parameters:
        simulation_params: dict
            A dictionary containing the parameters of the simulation.
        eigenstates: np.ndarray
            The eigenvectors of the Hamiltonian.
        n_state: int
            The index of the state to visualize.
        n_orbital: int
            The index of the orbital to visualize.
    
    Returns:
        fig: matplotlib.figure.Figure
            The figure of the probability
    '''

    norbs = 4 # Degrees of freedom
    
    # Extracting the n-th state eigenvector
    n_state_evec = eigenstates[:, n_state]

    Lx, Ly, Lz = simulation_params['L'], simulation_params['W'], simulation_params['H']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']

    # Reshape the eigenvector because there are multiple orbitals per site
    n_state_wavefunction = n_state_evec.reshape((int(Lx/ax - 1) * int(Ly/ay - 1) * int(Lz/az - 1), norbs))

    # Separate the wavefunction into individual orbital components, and extract orbital component n_orbital
    n_orbital_wavefunction = n_state_wavefunction[:, n_orbital].reshape(int(Lx/ax - 1), int(Ly/ay - 1), int(Lz/az - 1))

    # Visualize the probability density of the n-th state at different z-values, of state n_state, orbital n_orbital
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace=1.5)
    axes = axes.flatten()

    ymid = int(Ly/ay/2)

    maxi = np.max(np.abs(n_orbital_wavefunction)**2)
    
    for i, z in enumerate(np.linspace(0, Lz/az - 2, 3, dtype = int)):
        
        # Plot imshow of the probability density at different z-values
        image = np.abs(n_orbital_wavefunction[:, :, z].T)**2 / maxi
        p = axes[i].contourf(image, levels = 20)
        axes[i].contour(image, levels = 5, colors = 'k', linewidths = 0.8)
        axes[i].set(xlabel = r"$x$ [nm]", ylabel = r"$y$ [nm]", title = f'z = {z*az}nm')
        axes[i].set_aspect('equal')
        cb = fig.colorbar(p, ax = axes[i], orientation="horizontal", pad= 0.2, shrink=0.7)
        cb.ax.tick_params(rotation=45)
        cb.set_label(r"$|\psi(\mathbf{r})|^2 / |\psi_{\max}|^2$")
        
        # Change xticks and yticks by mulitplying by the lattice constant, so they are physically correct
        axes[i].set_xticks(np.arange(0, Lx/ax, 10))
        axes[i].set_xticklabels(np.arange(0, Lx, 10*ax))
        axes[i].set_yticks(np.arange(0, Ly/ay, 10))
        axes[i].set_yticklabels(np.arange(0, Ly, 10*ay))

        # Plot line plot of the probability density at different z-values
        axes[i+3].plot(np.abs(n_orbital_wavefunction[:, ymid, z])**2,
                       label = f"y = {ymid*ay} nm, z = {z*az} nm")
        axes[i+3].set_title("Probability density side view \n" + f"at height z = {z*az}nm")
        axes[i+3].set(xlabel =r"$x$ [nm]", ylabel = r"$|\psi|^2$")
        axes[i+3].legend(loc = 'lower center')

        # Change xticks by mulitpluing by the lattice constant, so they are physically correct
        axes[i+3].set_xticks(np.arange(0, Lx/ax, 10))
        axes[i+3].set_xticklabels(np.arange(0, Lx, 10*ax))

    #fig.suptitle(f"Parameters: Ly = {Ly}nm, Lx = {Lx}nm, Lz = {Lz}nm, B = {simulation_params['B']}T, V = {simulation_params['V1']}meV, s = {simulation_params['s1x']}nm" + 'n_state = ' + '\n' + str(n_state) + ', n_orbital = ' + str(n_orbital) )
    fig.tight_layout() 

    return fig, axes


def visualize_total_wavefunction_xy(simulation_params, eigenstates, n_state):
    ''' Visualize the total wavefunction at varying z-values'''

    norbs = 4                              # Degrees of freedom
    n_state_evec = eigenstates[:, n_state] # Extracting the n-th state eigenvector

    # Reshape the eigenvector because there are multiple orbitals per site
    n_state_wavefunction = n_state_evec.reshape(int(len(n_state_evec)/norbs), norbs)

    Lx, Ly, Lz = simulation_params['L'], simulation_params['W'], simulation_params['H']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']

    A = np.abs(n_state_wavefunction[:,0].reshape(int(Lx/ax-1), int(Ly/ay-1), int(Lz/az-1)))
    B = np.abs(n_state_wavefunction[:,1].reshape(int(Lx/ax-1), int(Ly/ay-1), int(Lz/az-1)))
    C = np.abs(n_state_wavefunction[:,2].reshape(int(Lx/ax-1), int(Ly/ay-1), int(Lz/az-1)))
    D = np.abs(n_state_wavefunction[:,3].reshape(int(Lx/ax-1), int(Ly/ay-1), int(Lz/az-1)))

    summi = A**2 + B**2 + C**2 + D**2
    maxi = np.max(summi) # for normalization
    
    fig, axes = plt.subplots(1, 3)
    axes = axes.flatten()

    for i,z in enumerate(np.linspace(1, int(Lz/az-2), 3, dtype = int)):
        image = summi[:, :, z].T / maxi
        p = axes[i].contourf(image, levels = 30, cmap = 'coolwarm')
        axes[i].contour(image, levels = 5, colors = 'k', linewidths = 0.8)
        axes[i].set(xlabel = r"$x$ [nm]", ylabel = r"$y$ [nm]", title = f'z = {z*az}nm')
        axes[i].set_aspect('equal')
        cb = fig.colorbar(p, ax = axes[i], orientation="horizontal", pad= 0.3, shrink=0.7)
        cb.ax.tick_params(rotation=45)
        cb.set_label(r"$|\psi(\mathbf{r})|^2 / |\psi_{\max}|^2$")
    
        # Change xticks and yticks by mulitplying by the lattice constant, so they are physically correct
        axes[i].set_xticks(np.arange(0, Lx/ax-1, 10))
        axes[i].set_xticklabels(np.arange(0, Lx-ax, 10*ax))
        axes[i].set_yticks(np.arange(0, Ly/ay-1, 10))
        axes[i].set_yticklabels(np.arange(0, Ly-ay, 10*ay))
        axes[i].tick_params(axis = 'x', labelrotation = 45)
    
    fig.suptitle(f"State {n_state}")
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.show()

def visualize_potential_xy(simulation_params, potential_function):
    ''' Visualize the potential at varying z-values'''

    Lx, Ly, Lz = simulation_params['L'], simulation_params['W'], simulation_params['H']
    ax, ay, az = simulation_params['ax'], simulation_params['ay'], simulation_params['az']

    x = np.arange(0, Lx - ax, ax)
    y = np.arange(0, Ly - ay, ay)
    z = np.arange(0, Lz - az, az)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the potential
    V = Gaussian_symmetric_double_dot(X, Y, Z)

    fig, axes = plt.subplots(1, 3)
    axes = axes.flatten()

    for i,z in enumerate(np.linspace(1, int(Lz/az-2), 3, dtype = int)):
        image = V[:, :, z].T
        p = axes[i].contourf(image, levels=30)
        axes[i].contour(image, levels=5, colors='k', linewidths=0.8)
        axes[i].set(xlabel=r"$x$ [nm]", ylabel=r"$y$ [nm]", title=f'z = {z*az}nm')
        axes[i].set_aspect('equal')
        cb = fig.colorbar(p, ax=axes[i], orientation="horizontal", pad=0.3, shrink=0.7)
        cb.ax.tick_params(rotation=45)
        cb.set_label(r"$V(\mathbf{r})$ [meV]")

        # Change xticks and yticks by multiplying by the lattice constant
        axes[i].set_xticks(np.arange(0, Lx/ax-1, 10))
        axes[i].set_xticklabels(np.arange(0, Lx-ax, 10*ax))
        axes[i].set_yticks(np.arange(0, Ly/ay-1, 10))
        axes[i].set_yticklabels(np.arange(0, Ly-ay, 10*ay))
        axes[i].tick_params(axis='x', labelrotation=45)

    fig.suptitle("Potential")
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.show()

