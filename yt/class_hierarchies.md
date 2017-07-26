# Developing `Line Plot`

ImagePlotContainer <- PlotWindow <- PWViewerMPL <- AxisAlignedSlicePlot
PlotMPL <- ImagePlotMPL <- WindowPlotMPL

- WindowPlotMPL: documented as "a container for a single `PlotWindow` matplotlib
  figure and axes"

Here's the road to `PlotMPL.__init__()`:

- `AxisAlignedSlicePlot.__init__()`. Explicitly call:
- `PWViewerMPL.__init__()` Explicitly call:
- `PlotWindow.__init__()` Call `self._setup_plots()`. `_setup_plots()` is not
  defined in `PlotWindow` but it is defined in the child `PWViewerMPL`
- `PWViewerMPL._setup_plots()`. Explicitly call `WindowPltMPL.__init__()`. So
  now we move from one of classes to the other.
- `WindowPlotMPL.__init__()`. Call `super().__init__()` which should go up to
  ImagePlotMPL...
- `ImagePlotMPL.__init__()`. Call `super.__init__()` which should go up to
  `PlotMPL`...
- `PlotMPL.__init__()`

Ok, so in python, you can call an object method from a parent that isn't defined
in the parent. This is different from C++ I believe.

So my question is where do fsize and axrect get set up.

`window_size` is a keyword argument to `AxisAlignedSlicePlot` and defaults to
`8.0`. `Aspect` is also a kwarg that defaults to `None`. Both of these get
passed as kwargs to the class constructor for `PWViewerMPL`. `PWViewerMPL`
doesn't do anything with those kwargs and passes them untouched along to
`PlotWindow`. `PlotWindow` has defaults for `aspect, window_size, and
buff_size`: `, None, 8.0, and (800,800)` respectively. `window_size` gets passed
as the second argument to `ImagePlotContainer.__init__()`. `aspect` and `buff_size` get
created as class data members without modification before `_setup_plots` is
called. Ok, looks like a default for `aspect` is set-up in
`_setup_plots`. `self.figure_size` is a class data member of PWViewerMPL, yet
it's not assigned anywhere in `plot_window.py` which contains the definitions
for `PWViewerMPL` and `PlotWindow`. Thus it must be in
`ImagePlotContainer`...Sure enough it's there and it's constructor takes
`figure_size` as the second argument. It's either a float or a tuple of two
floats. So `window_size` becomes `figure_size`. In `WindowPlotMPL` we have the
following important code block:
```python
        self._figure_size = figure_size

        # Compute layout
        fontscale = float(fontsize) / 18.0
        if fontscale < 1.0:
            fontscale = np.sqrt(fontscale)

        if iterable(figure_size):
            fsize = figure_size[0]
        else:
            fsize = figure_size
        self._cb_size = 0.0375*fsize
        self._ax_text_size = [1.2*fontscale, 0.9*fontscale]
        self._top_buff_size = 0.30*fontscale
        self._aspect = ((extent[1] - extent[0])/(extent[3] - extent[2])).in_cgs()
        self._unit_aspect = aspect

        size, axrect, caxrect = self._get_best_layout()
```

`_get_best_layout` is a method inherited from `ImagePlotMPL`. `extent` is an
argument to the constructor of `WindowPlotmPL` that is used to set its class
data member `self._aspect`.
