import yt
import numpy as np

ds = yt.load('/home/lindsayad/projects/moose/test/tests/bcs/nodal_normals/gold/cylinder_hexes_out.e', step=-1)
m = ds.index.meshes[0]
coords = m.connectivity_coords
indices = m.connectivity_indices
np.set_printoptions(threshold=np.inf)

fn = 'new_ds.txt'
with open(fn,'w') as f:
    f.write(repr(coords) + "\n")
    f.write(repr(indices))
#     f.write('[')
#     for i in range(coords.shape[0]):
#         f.write('[')
#         for j in range(coords.shape[1] - 1):
#             f.write(str(coords[i][j]) + ", ")
#         f.write(str(coords[i][coords.shape[1] - 1]))
#         f.write(']\n')
#     f.write(']\n\n')
#     f.write('[')
#     for i in range(indices.shape[0]):
#         f.write('[')
#         for j in range(indices.shape[1] - 1):
#             f.write(str(indices[i][j]) + ", ")
#         f.write(str(indices[i][indices.shape[1] - 1]))
#         f.write(']\n')
#     f.write(']\n')


# np.savetxt('coords.txt', coords)
# np.savetxt('indices.txt', indices)
# print(coords)
# print("\n")
# print(indices)
