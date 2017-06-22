import yt
ds = yt.load("/home/lindsayad/projects/zapdos/tests/TM10_circular_wg/gold/TM_steady_out.e", step=-1)
ad = ds.all_data()

from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 17
rcParams['font.serif'] = ['Computer Modern Roman']


field_list = [('all', 'Er'), ('all', 'Electric_z'), ('all', 'Hphi')]
field_label = {'Er' : r'E$_r$ (V/m)', 'Electric_z' : 'E$_z$ (V/m)', 'Hphi' : 'H$_\phi$ (A/m)'}

for field in field_list:
# field = field_list[0]
    field_type, field_name = field
    slc = yt.SlicePlot(ds, 'z', field)
    slc.set_log(field, False)
    slc.set_xlabel("r (mm)")
    slc.set_ylabel("z (mm)")
    slc.set_colorbar_label(field, field_label[field_name])
    slc.set_figure_size(1.2)
    slc.set_width((.015, .09))
    # slc.show()
    slc.save("/home/lindsayad/Pictures/tex_script" + field_name + ".png")
