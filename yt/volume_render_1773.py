import yt
from yt.testing import fake_hexahedral_ds

ds = fake_hexahedral_ds()

sc = yt.create_scene(ds)
cam = sc.camera
cam.focus = ds.arr([0.0, 0.0, 0.0], 'code_length')
cam_pos = ds.arr([-3.0, 3.0, -3.0], 'code_length')
north_vector = ds.arr([0.0, -1.0, -1.0], 'dimensionless')
cam.set_position(cam_pos, north_vector)

# increase the default resolution
cam.resolution = (800, 800)

sc.show()
