import yt

ds = yt.load("/home/lindsayad/projects/moose/modules/navier_stokes/tests/ins/lid_driven/gold/lid_driven_out.e", step=-1)
slc = yt.SlicePlot(ds, 'z', ('connect1', 'vel_y'))
slc.set_log(('connect1','vel_y'), False)
slc.set_width((1, 1))

slc.save()
