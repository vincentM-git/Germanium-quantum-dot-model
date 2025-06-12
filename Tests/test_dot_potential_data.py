import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

i = 2

data = np.load('../data/2QD_dataset_05_02_23_IST_10721_S14_device/self_consistent_potential_iterative/data{}.npy'.format(i))

#print(data)

nx, ny = data.shape

print(data.shape)

x1, x2, y1, y2 = 175, 280, 205, 270

data_red = data[y1:y2, x1:x2]

nyd, nxd = data_red.shape

print(len(data_red))

print((nxd, nyd))

# def onsite(site):
#     """
#     Converts potential energy data to python function using a scipy interpolator (possibly extrapolate)
#     """
#     xd, yd = np.arange(nxd), np.arange(nyd)

#     double_dot_potential = RegularGridInterpolator((xd, yd), data_red, bounds_error=False, fill_value=None)

#     return double_dot_potential(site)


plt.imshow(data)
plt.colorbar()
plt.show()

x = np.arange(nx)
y = np.arange(ny)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, data)
plt.show()

plt.imshow(data_red)
plt.colorbar()
plt.show()

x = np.arange(nxd)
y = np.arange(nyd)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, data_red)
plt.show()


xd, yd = np.arange(nxd), np.arange(nyd)

data_red = data_red.transpose()

double_dot_potential = RegularGridInterpolator((xd, yd), data_red, bounds_error=False, fill_value=None)

x = np.linspace(0, nxd, 10)
y = np.linspace(0, nyd, 10)

X, Y = np.meshgrid(x, y, indexing='ij')
Z = double_dot_potential((X,Y))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()