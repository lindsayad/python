import yt
ds = yt.load("yt_data/navier_stokes/RZ_p_parts_do_nothing_bcs_cone_out.e", step=-1)
ds.variables['vals_nod_var2'][:]
ad = ds.all_data()
ad
mesh = ds.index.meshes[0]
mesh
ad['vel_x'].max()
mesh.connectivity_coords
mesh.connectivity_coords.shape
mesh.connectivity_indices.shape
