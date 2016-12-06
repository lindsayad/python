import yt

data_dir = "MOOSE_sample_data"
# data_set = "out.e-s010"
# data_set = "mps_out.e"
data_set = "hex20-mesh_out.e"
# data_dir = "SecondOrderTets"
# data_set = "few-element-mesh_out.e"
data_dir2 = "SecondOrderTets"
data_set2 = "3d_unstructured_out.e"
ds = yt.load(data_dir + "/" + data_set, step=-1)
ds2 = yt.load(data_dir2 + "/" + data_set2, step=-1)
yt.interactive_render(ds)
# yt.interactive_render(ds, ("connect2", "temp"))
