import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sympy as sp
import pdb

def three_point_func(m, uniform_grid=False):
    x0, x1, x2, y0, y1, y2, x, h, h0, h1 = sp.symbols("x0 x1 x2 y0 y1 y2 x h h0 h1")
    if m == 0:
        x = x0
    if m == 1:
        x = x1
    if m == 2:
        x = x2
    fprime = y0 * ((2*x - x1 - x2) / ((x0 - x1) * (x0 - x2))) \
             + y1 * ((2*x - x0 - x2) / ((x1 - x0) * (x1 - x2))) \
             + y2 * ((2*x - x0 - x1) / ((x2 - x0) * (x2 - x1)))
    if uniform_grid:
        fprime = fprime.subs([(x1, x0 + h), (x2, x0 + 2*h)])
    else:
        fprime = fprime.subs([(x1, x0 + h0), (x2, x0 + h0 + h1)])
    print(sp.simplify(fprime))

def phi(t):
    return 3. * t**2 - 2. * t**3

def psi(t):
    return t**3 - t**2

def H1(xhi, xlo, x):
    h = xhi - xlo
    t = (xhi - x) / h
    return phi(t)

def H2(xhi, xlo, x):
    h = xhi - xlo
    t = (x - xlo) / h
    return phi(t)

def H3(xhi, xlo, x):
    h = xhi - xlo
    t = (xhi - x) / h
    return -h * psi(t)

def H4(xhi, xlo, x):
    h = xhi - xlo
    t = (x - xlo) / h
    return h * psi(t)

def p(xhi, xlo, fhi, flo, dhi, dlo, x):
    return flo * H1(xhi, xlo, x) + fhi * H2(xhi, xlo, x) \
        + dlo * H3(xhi, xlo, x) + dhi * H4(xhi, xlo, x)

def calc_derivs(xarr, farr, harr, delta):
    darr = np.zeros((xarr.size))
    for i in range(1, darr.size - 1):
        darr[i] = (harr[i-1]**2 * farr[i+1] - harr[i]**2 * farr[i-1] - farr[i] * (harr[i-1] - harr[i]) * (harr[i-1] + harr[i])) / (harr[i-1] * harr[i] * (harr[i-1] * harr[i]))
        # if (delta[i-1] * delta[i] <= 0):
        #     darr[i] = 0
        # else:
        #     gamma = (harr[i-1] + 2*harr[i]) / (3 * (harr[i-1] + harr[i]))
        #     darr[i] =  delta[i-1] * delta[i] / (gamma * delta[i] + (1 - gamma) * delta[i-1])
    darr[0] = (-harr[0]**2 * farr[2] - harr[1] * farr[0] * (2*harr[0] + harr[1]) + farr[1] * (harr[0] + harr[1])**2) / (harr[0] * harr[1] * (harr[0] + harr[1]))
    hlast = harr[harr.size - 1]
    hsecond = harr[harr.size - 2]
    flast = farr[farr.size - 1]
    fsecond = farr[farr.size - 2]
    fthird = farr[farr.size - 3]
    darr[darr.size - 1] = (hsecond * flast * (hsecond + 2 * hlast) + hlast**2 * fthird - fsecond * (hsecond + hlast)**2) / (hsecond * hlast * (hsecond + hlast))
    print(darr)
    return darr

def modify_derivs(alpha, beta, delta):
    tau = 3 / np.sqrt(alpha**2 + beta**2)
    alpha_star = alpha * tau
    beta_star = beta * tau
    dlo = alpha_star * delta
    dhi = beta_star * delta
    return dlo, dhi

def spline(xarr, farr):
    if (xarr.size != farr.size):
        RuntimeError("Independent and dependent variable arrays must "
                     "be the same length.")
    n_knots = xarr.size
    n_I = xarr.size - 1
    harr = np.zeros((n_I))
    for i in range(n_I):
        harr[i] = xarr[i+1] - xarr[i]
    delta = np.zeros((n_I))
    for i in range(n_I):
        delta[i] = (farr[i+1] - farr[i]) / harr[i]
    darr = calc_derivs(xarr, farr, harr, delta)
    alpha = np.zeros((n_I))
    beta = np.zeros((n_I))
    if np.sign(delta[0]) != np.sign(darr[0]):
        darr[0] = 0
    if np.sign(delta[n_I-1]) != np.sign(darr[n_knots-1]):
        darr[n_knots-1] = 0
    for i in range(n_I):
        # pdb.set_trace()
        # print("darr[i] equals {}; darr[i+1] equals {}; delta[i] equals {}.".format(darr[i], darr[i+1], delta[i]))
        if darr[i] == delta[i] == 0:
            alpha[i] = 1
        elif delta[i] == 0:
            alpha[i] = 4
        else:
            alpha[i] = darr[i] / delta[i]
        if darr[i+1] == delta[i] == 0:
            beta[i] = 1
        elif delta[i] == 0:
            beta[i] = 4
        else:
            beta[i] = darr[i+1] / delta[i]
        # print("alpha equals {} and beta equals {}".format(alpha[i], beta[i]))
        if alpha[i]**2 + beta[i]**2 > 9:
            # print("Damn, interpolation in region " + str(i) + " is not monotonic! Now fix it!")
            darr[i], darr[i+1] = modify_derivs(alpha[i], beta[i], delta[i])
    return darr

def sample(x, xarr, farr, darr):
    if (xarr.size != farr.size != darr.size):
        RuntimeError("All arrays must be the same length.")
    n_I = xarr.size - 1
    for i in range(n_I):
        if xarr[i] <= x <= xarr[i+1]:
            return p(xarr[i+1], xarr[i], farr[i+1], farr[i], darr[i+1], darr[i], x)


x = np.array([0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15])
y = np.array([10, 10, 10, 10, 10, 10, 10.5, 15, 50, 60, 85])
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = np.array([0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000])
# y = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
# x = np.array([0, 1, 2])
# y = np.array([0, 1, 8])
# y = np.array([0, 1, 4])
# x = np.array([0, 2, 4, 6, 8, 10])
# y = np.array([0, 8, 64, 216, 512, 1000])
# y = np.array([0, 4, 16, 36, 64, 100])
# n_knots = x.size
# n_I = x.size - 1
# harr = np.zeros((n_I))
# for i in range(n_I):
#     harr[i] = x[i+1] - x[i]
# print(calc_derivs(x, y, harr))
darr = spline(x,y)
print(darr)
# xnew = np.arange(0, 15.01, 0.01)
# xnew = np.arange(0, 10.01, 0.01)
# xnew = np.arange(0, 2.01, 0.01)
xnew = np.arange(0, 11, 1)
ynew = np.array([sample(xpt, x, y, darr) for xpt in xnew])
print(ynew)
# print(xnew[:10])
# print(ynew[:10])
# f = interp1d(x, y, kind='cubic')
# plt.plot(x, y, 'o', xnew, ynew, 'r-', xnew, f(xnew), 'b--')
# plt.plot(x, y, 'o', xnew, ynew, 'r-', xnew, xnew**3, 'b--')
# plt.plot(x, y, 'o', xnew, ynew, 'r-', xnew, xnew**2, 'b--')
# plt.show()


# # x = np.arange(12)
# # y = np.array([0, 1, 4.8, 6, 8, 13, 14, 15.5, 18, 19, 23, 24])
# # f = interp1d(x, y, kind='cubic')
# # xnew = np.arange(0, 11.1, 0.1)
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

# x = np.array([0, 1, 2, 3])
# y = np.array([0, 400, 400, 800])
# a = np.zeros((3))
# b = np.zeros((3))
# c = np.zeros((3))
# delta_x = np.array([x[i+1] - x[i] for i in range(x.size - 1)])
# delta_y = np.array([y[i+1] - y[i] for i in range(y.size - 1)])

# A_eq = np.zeros((10, 6))
# for i in range(3):
#     A_eq[3*i][2*i] += delta_x[i]**3
#     A_eq[3*i+1][2*i] += delta_x[i]**2
#     A_eq[3*i+2][2*i] += delta_x[i]
# for i in range(2):
#     A_eq[3*i][2*i+1] += 3*delta_x[i]**2
#     A_eq[3*i+1][2*i+1] += 2*delta_x[i]
#     A_eq[3*i+2][2*i+1] += 1
#     A_eq[3*i+5][2*i+1] += -1
