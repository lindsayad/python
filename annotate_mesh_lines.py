import yt

# We load the last time frame
ds = yt.load("/home/lindsayad/yt_data/MOOSE_sample_data/mps_out.e", step=-1,
             displacements={'connect2': (10.0, [0.01, 0.0, 0.0])})

# create a default scene
sc = yt.create_scene(ds, ("connect2", "temp"))

# override the default colormap. This time we also override
# the default color bounds
ms = sc.get_source(0)
ms.cmap = 'hot'
ms.color_bounds = (500.0, 1700.0)

# adjust the camera position and orientation
cam = sc.camera
camera_position = ds.arr([-1.0, 1.0, -0.5], 'code_length')
north_vector = ds.arr([0.0, -1.0, -1.0], 'dimensionless')
cam.width = ds.arr([0.05, 0.05, 0.05], 'code_length')
cam.set_position(camera_position, north_vector)

# increase the default resolution
cam.resolution = (800, 800)

# render, draw the element boundaries, and save
sc.render()
sc.annotate_mesh_lines()
sc.save()
