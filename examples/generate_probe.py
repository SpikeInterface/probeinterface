"""
This generate a probe from sratch.
"""


from probeinterface import Probe
from probeinterface.plotting import plot_probe
import numpy as np
import matplotlib.pyplot as plt


n = 24

positions = np.zeros((n, 2))
for i in range(n):
    x = i // 8
    y = i % 8
    positions[i] = x, y
positions *= 20
positions[8:16, 1] -= 10

probe = Probe(ndim=2, si_units='um')
probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 5})

probe.create_auto_shape(type='tip')


# and plot
plot_probe(probe)
plt.show()

