import yt

# data_dir = "MOOSE_sample_data"
# data_set = "out.e-s010"
data_dir = "SecondOrderTets"
data_set = "few-element-mesh_out.e"
# data_dir = "SecondOrderTets"
# data_set = "tet10_unstructured_out.e"
ds = yt.load(data_dir + "/" + data_set, step=-1)

# create a default scene
sc = yt.create_scene(ds)

# override the default colormap
ms = sc.get_source(0)
ms.cmap = 'Eos A'
ms.color_bounds = (.07, .15)
# ms.color_bounds = (-.01, .3)

# adjust the camera position and orientation
cam = sc.camera
# increase the default resolution
cam.resolution = (800, 800)

# cam.focus = [0, 0, 0]
# cam_pos = [1, 0, 0]
# north_vector = [0, 1, 0]
# cam.set_position(cam_pos, north_vector)
# cam_pos = [0, 1, 0]
# north_vector = [0, 0, 1]
# cam.set_position(cam_pos, north_vector)

cam.focus = ds.arr([0.5, 0.5, 0.5], 'code_length')

cam_pos = ds.arr([0.5, 0.5, -1.0], 'code_length')
north_vector = [1,0,0]
cam.set_position(cam_pos, north_vector)
sc.save(data_set + "_from-minus-z-axis-second-order-embree.png")

cam_pos = ds.arr([0.5, 0.5, 4.0], 'code_length')
north_vector = [1,0,0]
cam.set_position(cam_pos, north_vector)
sc.save(data_set + "_from-plus-z-axis-second-order-embree.png")

cam_pos = ds.arr([2, .5, .5], 'code_length')
north_vector = ds.arr([0.0, 1.0, 0.0], 'dimensionless')
cam.set_position(cam_pos, north_vector)
# sc.render()
# sc.annotate_mesh_lines()
sc.save(data_set + "_from-plux-x-axis-embree.png")

cam_pos = ds.arr([.5, 2, .5], 'code_length')
north_vector = ds.arr([0, 0, 1], 'dimensionless')
cam.set_position(cam_pos, north_vector)
sc.save(data_set + "_from-plux-y-axis-embree.png")

cam_pos = ds.arr([-1, .5, .5], 'code_length')
north_vector = ds.arr([0.0, 1.0, 0.0], 'dimensionless')
cam.set_position(cam_pos, north_vector)
sc.save(data_set + "_from-minus-x-axis-embree.png")

cam_pos = ds.arr([.5, -1, .5], 'code_length')
north_vector = ds.arr([0, 0, 1], 'dimensionless')
cam.set_position(cam_pos, north_vector)
sc.save(data_set + "_from-minus-y-axis-embree.png")
