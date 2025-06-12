import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Modules')))

from LK_utils import H_LKBP, get_material_parameters
from quantum_dots_LKBP import make_model_LKBP, eigs_double_dot_potential, D_dip, visualize_total_wavefunction_xy
import time

L = 240
W = 120
H = 20

# Parameter for definition of computation lattice

ax, ay, az = 4, 4, 1

lat_params = {"L": L, "W": W, "H": H, "ax": ax, "ay": ay, "az": az}
print('Lattice parameters (lengths in nm): {}'.format(lat_params))

nx, ny, nz = int(L/ax - 1), int(W/ay - 1), int(H/az - 1)
print('Number of points on the lattice: {0}*{1}*{2}={3}'.format(nx, ny, nz, nx*ny*nz))

print(H_LKBP)

mat = 'Ge'
mat_params = get_material_parameters(mat)
print('Material parameters: {}'.format(mat_params))

B = [0, 0, 0.01]

d, s, U0, eEz = 120, 50, 4, 0

# Strain parameters (ref: Abadillo PRL 2023)

eps_para = -0.006
eps_perp = 0.005

t1 = time.time()
print('Computing charge and spin splitting, and electric dipole moments for magnetic field B = {0} T and electric field = {1:.1f} mV/nm'.format(B, eEz))
syst = make_model_LKBP(H_LKBP, mat_params, lat_params, eps_para, eps_perp, B)
eigs = eigs_double_dot_potential(syst, lat_params, B, d, s, U0, eEz)
[Dc, Ds, x01, y01, z01, xl, yl, zl] = D_dip(eigs, lat_params)
t2 = time.time()

print('Dc = {0} meV, Ds = {1} meV'.format(Dc, Ds))
print('x01 = {0:.5f} nm'.format(x01))
print('y01 = {0:.5f} nm'.format(y01))
print('z01 = {0:.5f} nm'.format(z01))
print('xl = {0:.5f} nm'.format(xl))
print('yl = {0:.5f} nm'.format(yl))
print('zl = {0:.5f} nm'.format(zl))
print('Done in {0:3f} s'.format(t2-t1))

visualize_total_wavefunction_xy(simulation_params = lat_params,
                                   eigenstates = eigs[1],
                                     n_state=1)

