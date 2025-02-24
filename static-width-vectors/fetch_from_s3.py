import boto3
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib
from matplotlib import cm

# If you are using WSL to run this code, you need this line. Otherwise, comment it out.
matplotlib.use('TkAgg')

# Set the date stamp
DATE_STAMP = "24-02-25"
DOWNLOAD_RESULTS = True

if DOWNLOAD_RESULTS:
    # Open an S3 client and move the file using the date as a prefix
    s3 = boto3.resource('s3')
    print("Downloading result file from S3..")
    s3.meta.client.download_file('sve-results', f'{DATE_STAMP}/results.dat', './results.dat')

# Open the results file and graph some results using matplotlib
NX = 512
NY = 512
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

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(x_grid, y_grid, rho_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()