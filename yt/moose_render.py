import yt

ds = yt.load("MOOSE_sample_data/out.e-s010")

# create a default scene
sc = yt.create_scene(ds)

# override the default colormap
ms = sc.get_source(0)
ms.cmap = 'Eos A'

# adjust the camera position and orientation
cam = sc.camera
cam.focus = ds.arr([0.0, 0.0, 0.0], 'code_length')
cam_pos = ds.arr([-3.0, 3.0, -3.0], 'code_length')
north_vector = ds.arr([0.0, -1.0, -1.0], 'dimensionless')
cam.set_position(cam_pos, north_vector)

# increase the default resolution
cam.resolution = (800, 800)

# render and save
sc.save()
