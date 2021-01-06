"""
Generate a Probe from scatch
----------------------------

This generate a probe from sratch.
"""

##############################################################################

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe
from probeinterface.plotting import plot_probe


##############################################################################
# First, let's create dummy positions for 32 electrodes probe

n = 24
positions = np.zeros((n, 2))
for i in range(n):
    x = i // 8
    y = i % 8
    positions[i] = x, y
positions *= 20
positions[8:16, 1] -= 10

##############################################################################
# create a Probe object
# and set position and shape of each electrodes
# 
# the `ndim` means that the electrode is 2d so the position has shape (n_elec, 2)
# we can also define 3d probe with `ndim=3` and so positions will have shape (n_elec, 3)
# 
# Note: `shapes` and `shape_params` could be array as well

probe = Probe(ndim=2, si_units='um')
probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 5})

##############################################################################
# probe object have a fancy print

print(probe)

##############################################################################
# cerate the planar contour (polygon) of the probe

polygon = [(-20, -30), (20, -110), (60, -30), (60, 190), (-20, 190)]
probe.set_planar_contour(polygon)

##############################################################################
# if pandas is installed the probe object can be export to a dataframe for simpler view

df = probe.to_dataframe()
df

##############################################################################
# and plot it (need matmatplotlib installed)

plot_probe(probe)

##############################################################################
# We can transorm this Probe into 3d probe this.
# Here the 'y' will be 0 for all electrodes.

probe_3d = probe.to_3d(plane='xz')
plot_probe(probe_3d)


plt.show()

