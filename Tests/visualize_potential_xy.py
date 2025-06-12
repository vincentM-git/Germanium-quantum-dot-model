import numpy as np
import matplotlib.pyplot as plt
from double_dot import Gaussian_symmetric_double_dot

L = 300
W = 150
H = 20

# Parameter for definition of computation lattice

ax, ay, az = 4, 4, 1

s, d, U0, eEz = 50, 150, 4, 1

params = {"L": L, "W": W, "H": H, "ax": ax, "ay": ay, "az": az, 's': s, 'd': d, 'V0': U0, 'eEz': eEz}

def visualize_double_Gaussian_potential_xy(params):
    # Extract parameters
    L, W, H = params["L"], params["W"], params["H"]
    d, U0, s = params["d"], params["V0"], params["s"]

    # xy-plane at z = H/2
    x = np.linspace(-L/2, L/2, 200)
    y = np.linspace(-W/2, W/2, 200)
    X, Y = np.meshgrid(x, y)
    V_xy = Gaussian_symmetric_double_dot(X, Y, d, U0, s)

    # Potential along x at y=0
    V_x = Gaussian_symmetric_double_dot(x, 0, d, U0, s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 2D potential plot
    c = ax1.imshow(V_xy, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='equal')
    ax1.set_title('Double Gaussian potential energy (xy-plane)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    fig.colorbar(c, ax=ax1, label='Potential [meV]')

    # 1D cut at y=0
    ax2.plot(x, V_x)
    ax2.set_title('Potential energy at y=0')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('Potential [meV]')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

visualize_double_Gaussian_potential_xy(params)