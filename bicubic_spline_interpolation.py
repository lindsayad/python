from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 2)
y = np.arange(-5, 5, 2)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2+yy**2)
f = interpolate.interp2d(x, y, z, kind='cubic')

xnew = np.arange(-5, 5, 1)
ynew = np.arange(-5, 5, 1)
xxnew, yynew = np.meshgrid(xnew, ynew)
znew = f(xnew, ynew)
zcomp = np.sin(xxnew**2 + yynew**2)
# plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
# plt.show()
