import numpy as np

xks = np.array([0, 1, 0, 0, 0.5, 0.5, 0  , 0  , 0.5, 0])
yks = np.array([0, 0, 1, 0, 0  , 0.5, 0.5, 0  , 0  , 0.5])
zks = np.array([0, 0, 0, 1, 0  , 0  , 0  , 0.5, 0.5, 0.5])
coeff_array = np.zeros((xks.size, xks.size))
for i in range(xks.size):
    coeff_array[i][0] = 1
    coeff_array[i][1] = xks[i]
    coeff_array[i][2] = (xks[i])**2
    coeff_array[i][3] = yks[i]
    coeff_array[i][4] = (yks[i])**2
    coeff_array[i][5] = zks[i]
    coeff_array[i][6] = (zks[i])**2
    coeff_array[i][7] = xks[i] * yks[i]
    coeff_array[i][8] = xks[i] * zks[i]
    coeff_array[i][9] = zks[i] * yks[i]
print(coeff_array)    
inverse = np.linalg.inv(coeff_array)
identity = np.identity(xks.size)
soln = np.empty((0,xks.size))
for i in range(xks.size):
    soln = np.append(soln, np.array([np.matmul(inverse, identity[:,i])]), axis=0)
print(soln)

def shape(coeffs, x, y, z):
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * y + \
           coeffs[4] * y**2 + coeffs[5] * z + coeffs[6] * z**2 + \
           coeffs[7] * x * y + coeffs[8] * x * z + coeffs[9] * y * z

# for i in range(xks.size):
#     for j in range(xks.size):
#         print(shape(soln[i,:], xks[j], yks[j], zks[j]))
#     wait = input("PRESS ENTER TO CONTINUE")
