import yt
from yt.testing import fake_random_ds, assert_equal

def _test(field, data):
    return data[('stream', 'velocity_x')]

ds = fake_random_ds()
ds.add_field(('stream, density'), function=_test, units='cm/s', force_override=True)
assert_equal(ds.all_data()[('stream', 'density')], ds.all_data()[('stream', 'velocity_x')])
