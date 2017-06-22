import random
import numpy as np

import itertools

i = 0
coords = np.zeros((27, 3))
for combination in itertools.product(range(-1,2), repeat=3):
    for j in range(3):
        coords[i][j] = combination[j]
    i+=1

for i in range(27):
    for j in range(3):
        coords[i][j] = coords[i][j] + round(random.uniform(-.001,.001), 4)

fn = "test.geo"
with open(fn, 'w') as f:
    for i in range(27):
        f.write('Point({}) = {{{:.4f}, {:.4f}, {:.4f}}};\n'.format(i + 1, coords[i][0], coords[i][1], coords[i][2]))

np.set_printoptions(precision=4)
print(coords)
