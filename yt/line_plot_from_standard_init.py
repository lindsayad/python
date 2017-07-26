from os.path import expanduser
import yt

# home = expanduser("~")

# ds = yt.load(home + "/yt_data/SecondOrderTris/RZ_p_no_parts_do_nothing_bcs_cone_out.e", step=-1)
# ln = yt.LinePlot(ds, [('all', 'v'), ('all', 'u')], (0, 0, 0), (0, 1, 0), 1000, labels={('all', 'u') : r"u$_s$", ('all', 'v') : r"v$_s$"})
# # ln.add_plot([('all', 'v'), ('all', 'u')], (0, 0, 0), (1, 1, 0), 1000, labels={('all', 'u') : r"u$_l$", ('all', 'v') : r"v$_l$"})
# # ln.add_legend()
# # ln.set_xlabel("Arc Length (cm)")
# # ln.set_ylabel(r"Velocity (m s$^{-1}$)")
# ln.save("line_plot.eps")
# # slc = yt.SlicePlot(ds, 'z', ('all', 'u'))
# # slc.show()
# # slc.save()

def _unity(field, data):
    return 1.

ds = yt.load("/home/lindsayad/projects/moose/modules/navier_stokes/tests/ins/lid_driven/gold/lid_driven_out.e", step=-1)
ds.add_field(('all', 'vel_x'), function=_unity, units="m/s", force_override=True, take_log=False)
ds.add_field(('all', 'vel_y'), function=_unity, units="m/s", force_override=True, take_log=False)
ds.add_field(('all', 'T'), function=_unity, units="K", force_override=True, take_log=False)
ds.add_field(('all', 'p'), function=_unity, units="Pa", force_override=True, take_log=False)
ln = yt.LinePlot(ds, [('all', 'vel_x'), ('all', 'vel_y'), ('all', 'T'), ('all', 'p')], [0, 0.5, 0], [1, 0.5, 0], 100)
ln.annotate_legend(('all', 'vel_y'))
# import pdb; pdb.set_trace()
ln.save()
