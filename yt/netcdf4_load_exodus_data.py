# import yt
# ds = yt.load("RZ_p_no_parts_do_nothing_bcs_cone_out.e", step=-1)
# ds.meshes[0]
# ds.index
# ds.index.meshes[0]
# ds.index.meshes[0].connectivity_indices
# ds.index.meshes[0].connectivity_indices.shape[1]
# ds.index.meshes[0].connectivity_indices.shape[0]
# ds.index.meshes[0].connectivity_indices.shape
# ds.index.meshes[0].connectivity_coords.shape
# ds.variables
# ad = ds.all_data()
# ad['vel_x']
# ad['vel_x'].max()
# ad['vel_x'].min()
# import netCDF4
# import numpy as np
# ncdf_ds = netCDF4.Dataset('/home/lindsayad/moltres-submodule/tests/neutronics/gold/ne_deficient_b.e')
# # ncdf_ds = netCDF4.Dataset('/home/lindsayad/zapdos-submodule/tests/1d_dc/gold/mean_en_out.e')
# # ncdf_ds = netCDF4.Dataset('/home/lindsayad/moltres-submodule/tests/laminar_flow/gold/RZ_p_no_parts_do_nothing_bcs_cone_out.e')
# for key, value in ncdf_ds.variables.items():
#     # if "coord" in key:
#     print(key)
#     print(value)
#     # print(ncdf_ds.varables[key])
# print(ncdf_ds)
# print(ncdf_ds.variables['vals_nod_var1'][:])
# ds = yt.load('/home/lindsayad/moltres-submodule/tests/laminar_flow/gold/RZ_p_no_parts_do_nothing_bcs_cone_out.e')
# ds = yt.load('/home/lindsayad/moltres-submodule/tests/neutronics/gold/ne_deficient_b.e')
# print(ds.index.meshes[0].connectivity_coords.shape)
# ad = ds.all_data()
# print(ad[('connect1','u')])
import yt
ds = yt.load('ne_deficient_b.e')
ad = ds.all_data()
for key in ds.field_list:
    print(ad[key])
# print(ad[('connect1','u')])
