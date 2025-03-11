import boto3
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib
from matplotlib import cm

# If you are using WSL to run this code, you need this line. Otherwise, comment it out.
matplotlib.use('TkAgg')

# Open the results file and graph some results using matplotlib
NX = 1024
NY = 1024
data = np.genfromtxt('./results.dat')
x = data[:,0]
y = data[:,1]
rho = data[:,2]
ux = data[:,3]
uy = data[:,4]
T = data[:,5]
x_grid = x.reshape((NX, NY))
y_grid = y.reshape((NX, NY))
rho_grid = rho.reshape((NX, NY))
ux_grid = ux.reshape((NX, NY))
uy_grid = uy.reshape((NX, NY))
T_grid = T.reshape((NX, NY))

# Plot the surface.
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(x_grid, y_grid, T_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#plt.show()

# Contours of density
# fig, ax = plt.subplots(subplot_kw={"projection": "2d"})
plt.contour(x_grid, y_grid, rho_grid, levels=20)
plt.show()