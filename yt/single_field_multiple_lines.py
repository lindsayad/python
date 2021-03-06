import yt

def _unity(field, data):
    return 1.

ds = yt.load("/home/lindsayad/projects/moose/modules/navier_stokes/tests/ins/lid_driven/gold/lid_driven_out.e", step=-1)
ds.add_field(('all', 'vel_x'), function=_unity, units="m/s", force_override=True, take_log=False)
ds.add_field(('all', 'vel_y'), function=_unity, units="m/s", force_override=True, take_log=False)
ds.add_field(('all', 'T'), function=_unity, units="K", force_override=True, take_log=False)
ds.add_field(('all', 'p'), function=_unity, units="Pa", force_override=True, take_log=False)
lines = []
lines.append(yt.LineBuffer(ds, [0, 0.5, 0], [1, 0.5, 0], 100, label='y = 0.5'))
lines.append(yt.LineBuffer(ds, [0, 0.25, 0], [1, 0.25, 0], 100, label='y = 0.25'))
plt = yt.LinePlot.from_lines(ds, [('all', 'vel_x')], lines)
plt.annotate_legend(('all', 'vel_x'))
plt.save("single_field_multi_lines.png")
