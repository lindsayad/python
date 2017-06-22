import yt
from matplotlib.animation import FuncAnimation
from matplotlib import rc_context

ds = yt.load('simple_transient_diffusion_out.e', step=0)

plot = yt.SlicePlot(ds, 'z', ('all','u'))
plot.set_zlim(('all', 'u'), 0, 1)

fig = plot.plots[('all', 'u')].figure

# animate must accept an integer frame number. We use the frame number
# to identify which dataset in the time series we want to load

def animate(i):
    ds = yt.load('simple_transient_diffusion_out.e', step=i)
    plot._switch_ds(ds)

animation = FuncAnimation(fig, animate, frames=range(1, ds.num_steps))

# Override matplotlib's defaults to get a nicer looking font
with rc_context({'mathtext.fontset': 'stix'}):
    animation.save('animation_99.mp4', dpi=99)
