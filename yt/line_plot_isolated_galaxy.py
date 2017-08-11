import yt
from yt.units import meter, second

def _test(field, data):
    return data['velocity_x'] * meter / second

ds = yt.load('IsolatedGalaxy/galaxy0030/galaxy0030')
ds.add_field('density', function=_test, force_override=True, take_log=False)

plot = yt.LinePlot(ds, 'density', [0, 0, 0], [1, 1, 1], 512)

plot.save()
