"""
This generate a probe bunch from sratch.
"""
from probeinterface import Probe, ProbeBunch
from probeinterface.plotting import plot_probe_bunch
import numpy as np
import matplotlib.pyplot as plt




n = 24

positions = np.zeros((n, 2))
for i in range(n):
    x = i // 8
    y = i % 8
    positions[i] = x, y
positions *= 40
positions[8:16, 1] -= 10

# probe 0
probe0 = Probe(ndim=2, si_units='um')
probe0.set_electrodes(positions=positions, shapes='square', shape_params={'width': 18})
probe0.create_auto_shape(probe_type='tip')

# probe 1
probe1 = probe0.copy()
probe1.move([150, -50])

# probe bunch
probebunch = ProbeBunch()
probebunch.add_probe(probe0)
probebunch.add_probe(probe1)

#~ # and plot
#~ plot_probe_bunch(probebunch, separate_axes=False)
#~ plot_probe_bunch(probebunch, separate_axes=True)
#~ plt.show()


# 3D
probe0_3d = probe0.to_3d()
probe1_3d = probe1.to_3d()
probebunch = ProbeBunch()
probebunch.add_probe(probe0_3d)
probebunch.add_probe(probe1_3d)

plot_probe_bunch(probebunch, separate_axes=False)
plt.show()




