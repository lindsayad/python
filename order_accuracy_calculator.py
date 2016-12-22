import matplotlib.pyplot as plt
import numpy as np

def y(x):
    return x**2

def yp(x):
    return 2.*x

def num_deriv_fritz(x, h):
    return 2. * (y(x + h) - y(x)) * (y(x) - y(x - h)) \
         / (h * (y(x + h) - y(x - h)))

def lin_error(x, h):
    return 3. * 1. / h * (y(x) - y(x - h)) * 1. / h * (y(x + h) - y(x)) / (1. / h * (y(x + h) - y(x)) + 2. / h * (y(x) - y(x - h)))

def central_diff(x, h):
    return (y(x + h) - y(x - h)) / (2. * h)

def error(x, h, error_func):
    return np.abs(yp(x) - error_func(x, h))

x = 5.
h = np.arange(.01, 1.01, .1)
error_quad = np.array([error(x, hpt, num_deriv_fritz) for hpt in h])
error_lin = np.array([error(x, hpt, lin_error) for hpt in h])
error_central = np.array([error(x, hpt, central_diff) for hpt in h])
print(error_quad)
print(error_lin)
print(error_central)
plt.plot(h, error_quad, 'r-', h, error_lin, 'b--', h, error_central, 'k-')
# plt.legend()
plt.show()
