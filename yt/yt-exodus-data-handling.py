# from netCDF4 import Dataset
# ds = Dataset("RZ_p_no_parts_do_nothing_bcs_cone_out.e", step=-1)

# for i in [1,2,3]:
#     print(ds.variables['vals_nod_var%d' % i][-1, :].max())
#     print(ds.variables['vals_nod_var%d' % i][-1, :].min())


import yt
# import pdb; pdb.set_trace()

ds = yt.load("RZ_p_no_parts_do_nothing_bcs_cone_out.e", step=-1)
ad = ds.all_data()
print(ds.field_list)
# print(ad['vel_x'].max())

data = [ad['vel_x'], ad['vel_y'], ad['p']]
for datum in data:
    print(datum.max())
    print(datum.min())

# s1 = yt.SlicePlot(ds, 'z', 'vel_x')
# s1.set_log('all', False)
# s1.save("vel_x_2.png")

# s2 = yt.SlicePlot(ds, 'z', 'vel_y')
# s2.set_log('all', False)
# s2.save("vel_y.png")

# s3 = yt.SlicePlot(ds, 'z', 'p')
# s3.set_log('all', False)
# s3.save("p.png")

# s4 = yt.SlicePlot(ds, 'z', ('connect1', 'vertex_x'))
# s4.set_log('all', False)
# s4.save("vertex-x.png")

# # print(vel_x.max())
# # print(vel_y.max())
