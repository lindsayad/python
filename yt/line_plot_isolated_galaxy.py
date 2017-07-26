import yt

ds = yt.load('IsolatedGalaxy/galaxy0030/galaxy0030')

plot = yt.LinePlot(ds, 'density', [0, 0, 0], [1, 1, 1], 512)

plot.save()
