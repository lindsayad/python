from sage.all import *

x = var('x')
y = function('y')(x)

f = desolve(diff(y, x) - (1 - .25*x), y, ics=[0,0]); f

f.subs(x=6)

g = desolve(diff(y, x) - (-2 + .25*x), y, ics=[6, 1.5]); g

g.subs(x=8)

h = desolve(diff(y, x), y, ics=[8, 1]); h

import numpy as np

import matplotlib.pyplot as plt

pltx = np.linspace(0, 15, 16)
plty = np.zeros(16)
for i, val in enumerate(pltx):
    if val < 6:
        plty[i] = f.subs(x=val)
    elif 6 <= val < 8:
        plty[i] = g.subs(x=val)
    else:
        plty[i] = h.subs(x=val)

plt.plot(pltx, plty, label='analytic')

import yt

supgds = yt.load('/home/lindsayad/projects/moltres/problems/supg_on_source_hill.e', step=-1)
bufsupg = yt.LineBuffer(supgds, (0, 0, 0), (15, 0, 0), 16)


nosupgds = yt.load('/home/lindsayad/projects/moltres/problems/no_supg_on_source_hill.e', step=-1)
bufnosupg = yt.LineBuffer(nosupgds, (0, 0, 0), (15, 0, 0), 16)

bufnosupg[('all', 'u')]
bufsupg[('all', 'u')]
plt.plot(bufsupg.points.to_ndarray(), bufsupg[('all', 'u')].to_ndarray(), label='SUPG', linestyle='None', marker='o')
plt.plot(bufnosupg.points.to_ndarray(), bufnosupg[('all', 'u')].to_ndarray(), label='no SUPG', linestyle='None', marker='o')
plt.legend()
plt.savefig('/home/lindsayad/Pictures/source_supg_effect_hill.png')

plt.show()
