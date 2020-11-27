"""
More plotting examples
------------------------------------

Here some examples to showcase plotting scenario.

"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeBunch
from probeinterface.plotting import plot_probe, plot_probe_bunch
from probeinterface import generate_multi_columns_probe, generate_linear_probe



##############################################################################
# 

fig, ax = plt.subplots()

probe0 = generate_multi_columns_probe()
plot_probe(probe0, ax=ax)

# make color for each probe
probe1 = generate_linear_probe(num_elec=9)
probe1.move([200, 0])
plot_probe(probe1, ax=ax, 
            electrode_colors=['red', 'cyan', 'yellow']*3)

# prepare yourself for carnaval
probe2 = generate_linear_probe()
probe2.move([400, 0])
n = probe2.get_electrode_count()
rand_colors = np.random.rand(n,3)
plot_probe(probe2, ax=ax, electrode_colors=rand_colors, 
            probe_shape_kwargs={ 'facecolor':'purple', 'edgecolor':'k', 'lw':0.5, 'alpha':0.2})

# and make alien probes
probe3 = Probe()
positions = [[0, 0], [0, 50], [25, 77], [45, 27]]
shapes = ['circle', 'square', 'rect', 'circle']
params = [{'radius': 10}, {'width': 30}, {'width': 20, 'height':12}, {'radius': 13}]
probe3.set_electrodes(positions=positions, shapes=shapes,
            shape_params=params)
probe3.create_auto_shape(probe_type='rect')
probe3.move([600, 0])
plot_probe(probe3, ax=ax, electrode_colors=['b', 'c', 'g', 'y'])



plt.show()





