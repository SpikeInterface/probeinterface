"""
Probe 2d and Probe 3d
-----------------------------------

This show manipulation of the probe in 2d or 3d
"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe
from probeinterface.plotting import plot_probe


##############################################################################
# First, let's create one 2d probe (32 electrodes)
# 

n = 24
positions = np.zeros((n, 2))
for i in range(n):
    x = i // 8
    y = i % 8
    positions[i] = x, y
positions *= 20
positions[8:16, 1] -= 10

probe_2d = Probe(ndim=2, si_units='um')
probe_2d.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 5})
probe_2d.create_auto_shape(probe_type='tip')

##############################################################################
# Lets tranform is into 3d probe
# 
# Here the plane is 'xy' so y will be 0 for all electrodes.
# The shape of probe_3d.electrode_positions is now 3

probe_3d = probe_2d.to_3d(plane='xz')
print(probe_2d.electrode_positions.shape)
print(probe_3d.electrode_positions.shape)


##############################################################################
#  the plotting switch to 3d

plot_probe(probe_3d)

##############################################################################
# we can create other probe in other axis


other_3d = probe_2d.to_3d(plane='yz')
plot_probe(other_3d)


##############################################################################
# Probe can be moved and rotated

probe_3d.move([0, 30, -50])
probe_3d.rotate(theta=35, center=[0,0, 0], axis=[0,1,1])

plot_probe(probe_3d)

plt.show()






