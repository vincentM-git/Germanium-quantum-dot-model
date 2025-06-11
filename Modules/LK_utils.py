import numpy as np
from scipy.constants import physical_constants
from numpy import matmul
from scipy.constants import physical_constants
import kwant
from material_parameters import parameters

# Get physical constants (all lengths in nm, energies in meV and magnetic field in T)

# Bohr radius in nm

a0 = 1e9*physical_constants["Bohr radius"][0]

# Hartree energy in meV

EH = 1e3*physical_constants["Hartree energy in eV"][0]

# Bohr magneton in meV/T

muB = 1e3*physical_constants["Bohr magneton in eV/T"][0]

# magnetic flux h/e in Tesla.nm^2

phi0 = 2e18*physical_constants["mag. flux quantum"][0]

# parameter in k.p Hamiltonian

mu = 0.5 * EH * a0**2

consts = { "mu": mu, "muB": muB, "phi0": phi0}

# Spin-3/2 matrices definition

sqrt3 = np.sqrt(3)

id4 = np.identity(4)

Jx = np.array(
    [[0, sqrt3/2, 0, 0], 
     [sqrt3/2, 0, 1, 0], 
     [0, 1, 0, sqrt3/2], 
     [0, 0, sqrt3/2, 0]])

Jy = np.array(
    [[0, -1j*sqrt3/2, 0, 0], 
     [1j*sqrt3/2, 0, -1j, 0], 
     [0, 1j, 0, -1j*sqrt3/2], 
     [0, 0, 1j*sqrt3/2, 0]])

Jz = np.array(
    [[3/2, 0, 0, 0], 
     [0, 1/2, 0, 0], 
     [0, 0, -1/2, 0], 
     [0, 0, 0, -3/2]])

Jyz = matmul(Jy, Jz) + matmul(Jz, Jy)
Jzx = matmul(Jz, Jx) + matmul(Jx, Jz)
Jxy = matmul(Jx, Jy) + matmul(Jy, Jx)
Jx2 = matmul(Jx, Jx)
Jy2 = matmul(Jy, Jy)
Jz2 = matmul(Jz, Jz)
Jx3 = matmul(Jx2, Jx)
Jy3 = matmul(Jy2, Jy)
Jz3 = matmul(Jz2, Jz)

Jmatrices = {"id4": id4, "Jx": Jx, "Jy": Jy, "Jz": Jz,
            "Jx2": Jx2, "Jy2": Jy2, "Jz2": Jz2,
            "Jyz": Jyz, "Jzx": Jzx, "Jxy": Jxy,
            "Jx3": Jx3, "Jy3": Jy3, "Jz3": Jz3}

# Luttinger-Kohn-Bir-Pikus Hamiltonian definition

H_LKBP = ("""
    mu * ((ga1 + 5*ga2/2) * (k_x**2 + k_y**2 + k_z**2 + 2 * k_x * Ax(y,z) + 2 * k_y * Ay(x,z)) * id4 
    - 2 * ga2 * (k_x**2 * Jx2 + k_y**2 * Jy2 + k_z**2 * Jz2 + 2 * k_x * Ax(y,z) * Jx2 + 2 * k_y * Ay(x,z) * Jy2) 
    - ga3 * (kyz * Jyz + kzx * Jzx + kxy * Jxy 
    + (k_z * Ay(x,z) + Ay(x,z) * k_z) * Jyz + (k_z * Ax(y,z) + Ax(y,z) * k_z) * Jzx 
    + (k_x * Ay(x,z) + Ax(y,z) * k_y + k_y * Ax(y,z) + Ay(x,z) * k_x) * Jxy)) 
    + 2 * muB * (ka * (Jx*Bx + Jy*By + Jz*Bz) + qu * (Jx3*Bx + Jy3*By + Jz3*Bz)) 
    + V(x, y, z) * id4
    + bv * (epsxx * Jx2 + epsyy * Jy2 + epszz * Jz2)
    """)

def get_material_parameters(mat):
    for row in parameters:
        if row["Material"] == mat:
            return row
