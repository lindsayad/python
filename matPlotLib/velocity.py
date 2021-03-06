#!/usr/bin/python
#
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc
rc('text', usetex=True)
#
mpl.rcParams.update({'font.size': 13})
#
arrowShrinkSize = 0.1
figureFontSize = 14
arrowWidth = 1
N_A = 6.02e23
DishRadius = 0.03
WaterHeight = 0.0035368
GapDistance = 0.0064632
NeedleRadius = .00062
#
fig = plt.figure()
ax = fig.gca()
#
LiqMagVelocityData = np.loadtxt('Liquid_velocity_magnitude.txt')
i = 0
a,b = LiqMagVelocityData.shape
ainitial = a
while i < a:
    if np.isnan(LiqMagVelocityData[i,2]):
	    LiqMagVelocityData = np.delete(LiqMagVelocityData, (i), axis = 0)
	    a,b = LiqMagVelocityData.shape
    else:
        i = i + 1		
#
afinal = a
#
Nx1, idx1 = np.unique(LiqMagVelocityData[:,0], return_inverse=True)
Nx2, idx2 = np.unique(LiqMagVelocityData[:,1], return_inverse=True)
#
LiqMagVelocity = np.empty((Nx1.shape[0],Nx2.shape[0]))
LiqMagVelocity[idx1,idx2] = LiqMagVelocityData[:,2]
#
r3, z3 = np.meshgrid(Nx1, Nx2)
#
LiqMagVelocity_min, LiqMagVelocity_max = np.min(LiqMagVelocity), np.max(LiqMagVelocity)
#
GasMagVelocityData = np.loadtxt('Gas_velocity_magnitude.txt')
i = 0
a,b = GasMagVelocityData.shape
ainitial = a
while i < a:
    if np.isnan(GasMagVelocityData[i,2]):
	    GasMagVelocityData = np.delete(GasMagVelocityData, (i), axis = 0)
	    a,b = GasMagVelocityData.shape
    else:
        i = i + 1		
#
afinal = a
#
Nx1, idx1 = np.unique(GasMagVelocityData[:,0], return_inverse=True)
Nx2, idx2 = np.unique(GasMagVelocityData[:,1], return_inverse=True)
#
GasMagVelocity = np.empty((Nx1.shape[0],Nx2.shape[0]))
GasMagVelocity[idx1,idx2] = GasMagVelocityData[:,2]
#
r4, z4 = np.meshgrid(Nx1, Nx2)
#
GasMagVelocity_min, GasMagVelocity_max = np.min(GasMagVelocity), np.max(GasMagVelocity)
#
LiqMagVelocityAx = plt.pcolor(r3, z3, LiqMagVelocity.transpose(), cmap = 'jet', vmin= 0, vmax = LiqMagVelocity_max, rasterized = True)
LiqMagVelocityBar = plt.colorbar()
LiqMagVelocityBar.set_label('Liquid velocity magnitude (m/s)')
GasMagVelocityAx = plt.pcolor(r4, z4, GasMagVelocity.transpose(), cmap = 'jet', vmin= 0, vmax = GasMagVelocity_max, rasterized = True)
GasMagVelocityBar = plt.colorbar()
GasMagVelocityBar.set_label('Gas velocity magniutude (m/s)')
plt.axis([0,DishRadius+0.001,-WaterHeight,DishRadius-WaterHeight])
#
plt.xlabel('r (m)',fontsize = figureFontSize)
plt.ylabel('z (m)', fontsize = figureFontSize)
#
plt.plot([DishRadius, DishRadius],[-WaterHeight,GapDistance], color='k', linestyle = '-', linewidth = 2)
plt.plot([NeedleRadius,NeedleRadius],[GapDistance,DishRadius-WaterHeight], color='k', linestyle = '-', linewidth = 2)
plt.plot([NeedleRadius+1e-4,NeedleRadius+1e-4],[GapDistance,DishRadius-WaterHeight], color='k', linestyle = '-', linewidth = 2)
plt.plot([NeedleRadius+2e-4,NeedleRadius+2e-4],[GapDistance,DishRadius-WaterHeight], color='k', linestyle = '-', linewidth = 2)
plt.plot([0,DishRadius],[0,0], color='k', linestyle = '-', linewidth = 2) 
#
ax.text(0.015, 0.0003, 'Gas-liquid interface',color = 'r')
# 
VelocityData = np.loadtxt('Gas_velocity_arrows.txt')  
i = 0
a,b = VelocityData.shape
ainitial = a
while i < a:
    if np.isnan(VelocityData[i,2]):
	    VelocityData = np.delete(VelocityData, (i), axis = 0)
	    a,b = VelocityData.shape
    else:
        i = i + 1
#
amid1 = a
#
Nx1, idx1 = np.unique(VelocityData[:,0], return_inverse=True)
Nx2, idx2 = np.unique(VelocityData[:,1], return_inverse=True)
#
RVelocity = np.empty((Nx1.shape[0],Nx2.shape[0]))
RVelocity[idx1,idx2] = VelocityData[:,2]
ZVelocity = np.empty((Nx1.shape[0],Nx2.shape[0]))
ZVelocity[idx1,idx2] = VelocityData[:,3]
#
r, z = np.meshgrid(Nx1, Nx2)
Q = quiver(r, z, RVelocity.transpose(), ZVelocity.transpose(), angles = 'xy', scale = 1.0, color = 'r')
qk = quiverkey(Q, 0.003,0.027,0.1,"0.1 m/s (gas)",coordinates='data',color='k')
LiquidVelocityData = np.loadtxt('Liquid_velocity_arrows.txt')  
i = 0
a,b = LiquidVelocityData.shape
ainitial = a
while i < a:
    if np.isnan(LiquidVelocityData[i,2]):
	    LiquidVelocityData = np.delete(LiquidVelocityData, (i), axis = 0)
	    a,b = LiquidVelocityData.shape
    else:
        i = i + 1
#
amid1 = a
#
Nx1, idx1 = np.unique(LiquidVelocityData[:,0], return_inverse=True)
Nx2, idx2 = np.unique(LiquidVelocityData[:,1], return_inverse=True)
#
LiqRVelocity = np.empty((Nx1.shape[0],Nx2.shape[0]))
LiqRVelocity[idx1,idx2] = LiquidVelocityData[:,2]
LiqZVelocity = np.empty((Nx1.shape[0],Nx2.shape[0]))
LiqZVelocity[idx1,idx2] = LiquidVelocityData[:,3]
#
r2, z2 = np.meshgrid(Nx1, Nx2)
Q2 = quiver(r2, z2, LiqRVelocity.transpose(), LiqZVelocity.transpose(), angles = 'xy', scale = 0.1, color = 'r')
qk = quiverkey(Q2, 0.023,0.027,0.01,"0.01 m/s (liquid)",coordinates='data',color='k')
fig.savefig('velocity.eps',format='eps')
plt.show()
