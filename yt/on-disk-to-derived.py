import yt
from yt.units import meter, second

def _test(field, data):
    return data[('all', 'v')]

ds = yt.load('SecondOrderTris/RZ_p_no_parts_do_nothing_bcs_cone_out.e', step=-1)
ds.add_field(('all', 'u'), function=_test, force_override=True, take_log=False)
u = ds.all_data()[('all', 'u')]
v = ds.all_data()[('all', 'v')]
# u = ds.field_info[('all', 'u')]
# print(u(ds.all_data()))
