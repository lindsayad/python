import numpy as np
import yt
from yt.testing import fake_tetrahedral_ds
# from yt.testing import small_fake_hexahedral_ds as fake_hexahedral_ds
from yt.testing import fake_hexahedral_ds

def actual_center_and_widths(ds):
    actual_domain_widths = (ds.domain_right_edge.to_ndarray() - ds.domain_left_edge.to_ndarray()) / 1.2
    actual_le = ds.domain_left_edge.to_ndarray() + actual_domain_widths * .1
    actual_re = ds.domain_right_edge.to_ndarray() - actual_domain_widths * .1
    actual_center = (actual_le + actual_re) / 2
    return actual_domain_widths, actual_center

ds = fake_hexahedral_ds()
actual_domain_widths, actual_center = actual_center_and_widths(ds)
slc = yt.SlicePlot(ds, 'z', ('all', 'test'), center=actual_center, origin='native')
slc.set_width((actual_domain_widths[0],actual_domain_widths[1]))
slc.annotate_mesh_lines()
# slc.annotate_line((-3.7199e-1, 1.38729), (-5.60845e-2, 1.169447), coord_system='axis', plot_args={'color' : 'black'})
# slc.annotate_line((-3.7199e-1, 1.38729, .1), (-5.60845e-2, 1.1, .1), coord_system='data', plot_args={'color' : 'black'})
# slc.annotate_line((-.7, -.3, 0), (0, 0, 0), coord_system='data', plot_args={'color' : 'black'})
slc.save("new_ds.png")

# ds = yt.load("simple_diffusion_out.e", step=-1)
# actual_domain_widths, actual_center = actual_center_and_widths(ds)
# # slc = yt.SlicePlot(ds, 'z', ('all', 'u'), center=actual_center, origin='native')
# slc = yt.SlicePlot(ds, 'z', ('all', 'u'), center=(0, 0, 1), origin='native')
# slc.set_width((actual_domain_widths[0],actual_domain_widths[1]))
# slc.annotate_mesh_lines()
# slc.save()
