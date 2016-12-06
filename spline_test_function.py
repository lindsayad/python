import numpy as np
from scipy import interpolate

def u_func(x1,x2):
    a = 4
    b = 5
    c1 = 12
    c2 = 24
    c3 = 3
    c4 = 4
    c5 = 5
    c6 = 6
    return c1*(x1**4/12 - a*x1**3/6) + c2*(x2**4/12 - b*x2**3/6) \
        + c3*x1*x2 + c4*x1 + c5*x2 + c6

def cubic_func(x1,x2):
    return x1**3 + 2*x2**3

x1 = np.array([0, 2, 4])
x2 = np.array([0, 2, 4])
y = np.zeros((len(x1),len(x2)))

for foo,i in zip(x1, range(len(x1))):
    for bar,j in zip(x2, range(len(x2))):
        y[i][j] = cubic_func(foo, bar)
        # print(y[i][j])



# for foo,i in zip(x1, range(len(x1))):
#     for bar,j in zip(x2, range(len(x2))):
#         y[i][j] = u_func(foo, bar)

# f = interpolate.interp2d(x1, x2, y, kind='cubic')

x1new = np.arange(0, 5, 1)
x2new = np.arange(0, 5, 1)
for foo in x1new:
    for bar in x2new:
        print(cubic_func(foo, bar))
# for foo,i in zip(x1new, range(len(x1new))):
#     for bar,j in zip(x2new, range(len(x2new))):
#         ynew[i][j] = u_func(foo, bar)


# print(y)
# print()
# print(ynew)
# print()
# print(f(x1new, x2new))
