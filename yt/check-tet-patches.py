
# coding: utf-8

# In[1]:

import numpy as np
import yt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs


# In[3]:

ds = yt.load('SecondOrderTets/few-element-mesh_out.e', step=-1)


# In[4]:

index = 0  # selects an element
m = ds.index.meshes[0]
coords = m.connectivity_coords
indices = m.connectivity_indices - 1
vertices = coords[indices[index]]


# In[5]:

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

axis = [1, 1, 1]
theta = 1.2

# vertices -= vertices.mean(axis=0)
# vertices *= 5e3

for i in range(10):
    vertices[i] = np.dot(rotation_matrix(axis,theta), vertices[i])


# In[6]:

def patchSurfaceFunc(u, v, verts):
    return (1.0 - 3.0*u + 2.0*u*u - 3.0*v + 2.0*v*v + 4.0*u*v)*verts[0] + \
             (-u + 2.0*u*u)*verts[1] + \
             (-v + 2.0*v*v)*verts[2] + \
             (4.0*u - 4.0*u*u - 4.0*u*v)*verts[3] + \
             (4.0*u*v)*verts[4] + \
             (4.0*v - 4.0*v*v - 4.0*u*v)*verts[5]



# In[7]:

faces = [[0, 1, 3, 4, 8, 7],
         [2, 3, 1, 9, 8, 5],
         [0, 3, 2, 7, 9, 6],
         [0, 2, 1, 6, 5, 4]]


# In[8]:

# def plot_face(face, ax):
#     x = np.empty(15)
#     y = np.empty(15)
#     z = np.empty(15)
#     index = 0
#     for i, u in enumerate(np.arange(0.0, 1.25, 0.25)):
#         for j, v in enumerate(np.arange(0.0, 1.25 - u, 0.25)):
#             x[index], y[index], z[index] = patchSurfaceFunc(u, v, vertices[face])
#             # print('Mapped space of u=%s and v=%s' % (u, v))
#             # print('Corresponds to physical space of (%s, %s, %s)\n' % (x[index], y[index], z[index]))
#             index += 1
#     ax.clear()
#     ax.plot_trisurf(x, y, z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for k, face in enumerate(faces):
    # plot_face(face, ax)
    x = np.empty(15)
    y = np.empty(15)
    z = np.empty(15)
    index = 0
    for i, u in enumerate(np.arange(0.0, 1.25, 0.25)):
        for j, v in enumerate(np.arange(0.0, 1.25 - u, 0.25)):
            x[index], y[index], z[index] = patchSurfaceFunc(u, v, vertices[face])
            # print('Mapped space of u=%s and v=%s' % (u, v))
            # print('Corresponds to physical space of (%s, %s, %s)\n' % (x[index], y[index], z[index]))
            index += 1
    ax.plot_trisurf(x, y, z)
plt.show()

# x = np.empty(15)
# y = np.empty(15)
# z = np.empty(15)
# index = 0
# for i, u in enumerate(np.arange(0.0, 1.25, 0.25)):
#     for j, v in enumerate(np.arange(0.0, 1.25 - u, 0.25)):
#         x[index], y[index], z[index] = patchSurfaceFunc(u, v, vertices[faces[2]])
#         # print('Mapped space of u=%s and v=%s' % (u, v))
#         # print('Corresponds to physical space of (%s, %s, %s)\n' % (x[index], y[index], z[index]))
#         index += 1
# ax.plot_trisurf(x, y, z)
# plt.show()


    # plt.savefig("surface" + str(k) + ".png", format='png')
        # k += 1
# plt.show()
# In[ ]:
