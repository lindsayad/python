import numpy as np
import sympy as sp

def patchSurfaceFunc(vert0, vert1, vert2, vert3, vert4, vert5, vert6, vert7, u, v):

          S = 0.25*(1.0 - u)*(1.0 - v)*(-u - v - 1)*vert0 + \
                 0.25*(1.0 + u)*(1.0 - v)*( u - v - 1)*vert1 + \
                 0.25*(1.0 + u)*(1.0 + v)*( u + v - 1)*vert2 + \
                 0.25*(1.0 - u)*(1.0 + v)*(-u + v - 1)*vert3 + \
                 0.5*(1 - u)*(1 - v*v)*vert4 + \
                 0.5*(1 - u*u)*(1 - v)*vert5 + \
                 0.5*(1 + u)*(1 - v*v)*vert6 + \
                 0.5*(1 - u*u)*(1 + v)*vert7
          return S

vert0, vert1, vert2, vert3, vert4, vert5, vert6, vert7 = sp.symbols('vert0 vert1 vert2 vert3 vert4 vert5 vert6 vert7')

coordsu = [-1, 0, 1]
coordsv = [-1, 0, 1]
for u in coordsu:
    for v in coordsv:
        print(patchSurfaceFunc(vert0, vert1, vert2, vert3, vert4, vert5, vert6, vert7, u, v))
