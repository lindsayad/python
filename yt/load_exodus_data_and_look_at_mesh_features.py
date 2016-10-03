import yt
ds = yt.load("/home/lindsayad/data-yt/MOOSE_sample_data/RZ_p_no_parts_do_nothing_bcs_cone_out.e", step=-1)
# ds.variables['vals_nod_var2'][:]
ad = ds.all_data()
mesh = ds.index.meshes[0]
# print(ad['vel_x'].max())
# print(mesh.connectivity_coords)
# print(mesh.connectivity_coords.shape)
# print(mesh.connectivity_indices.shape)
