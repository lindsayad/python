import yt
from yt.units import meter, second

def _test(field, data):
    return data['velocity_x'] * meter / second

ds = yt.load('IsolatedGalaxy/galaxy0030/galaxy0030')
ds.add_field(('enzo', 'Density'), function=_test, units='cm**2/s**2', force_override=True, take_log=False)
ds.all_data()[('enzo', 'Density')]
plot = yt.LinePlot(ds, ('enzo', 'Density'), [0, 0, 0], [1, 1, 1], 512)

plot.save()
