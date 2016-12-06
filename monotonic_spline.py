import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# x = np.arange(12)
# y = np.array([0, 1, 4.8, 6, 8, 13, 14, 15.5, 18, 19, 23, 24])
# f = interp1d(x, y, kind='cubic')
# xnew = np.arange(0, 11.1, 0.1)
# plt.plot(x, y, 'o', xnew, f(xnew), '-')
# plt.show()

# x = np.array([0, 1, 2, 3])
# y = np.array([0, 400, 400, 800])
# f = interp1d(x, y, kind='cubic')
# xnew =  np.arange(0, 3.1, 0.1)
# plt.plot(x, y, 'o', xnew, f(xnew), '-')
# plt.show()

# x = np.array([0, 1, 1.1])
# y = np.array([0, 400, 410])
# f = interp1d(x, y, kind='cubic')
# xnew =  np.arange(0, 1.15, 0.05)
# plt.plot(x, y, 'o', xnew, f(xnew), '-')
# plt.show()

# c = [-1, 4]
# A = [[-3, 1], [1, 2]]
# b = [6, 4]
# x0_bounds = (None, None)
# x1_bounds = (-3, None)
# from scipy.optimize import linprog
# res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options={"disp": True})

x = np.array([0, 1, 2, 3])
y = np.array([0, 400, 400, 800])
a = np.zeros((3))
b = np.zeros((3))
c = np.zeros((3))
delta_x = np.array([x[i+1] - x[i] for i in range(x.size - 1)])
delta_y = np.array([y[i+1] - y[i] for i in range(y.size - 1)])

A_eq = np.zeros((10, 6))
for i in range(3):
    A_eq[3*i][2*i] += delta_x[i]**3
    A_eq[3*i+1][2*i] += delta_x[i]**2
    A_eq[3*i+2][2*i] += delta_x[i]
for i in range(2):    
    A_eq[3*i][2*i+1] += 3*delta_x[i]**2
    A_eq[3*i+1][2*i+1] += 2*delta_x[i]
    A_eq[3*i+2][2*i+1] += 1
    A_eq[3*i+5][2*i+1] += -1
