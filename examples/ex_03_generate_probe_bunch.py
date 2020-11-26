"""
Generate a ProbeBunch
-----------------------------------

This code show how to assmble several Probe into a ProbeBunch object.

"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeBunch
from probeinterface.plotting import plot_probe_bunch
from probeinterface import generate_fake_probe


##############################################################################
# Generate 2 fake Probe with util function
# 

probe0 = generate_fake_probe(elec_shapes='square')
probe1 = generate_fake_probe(elec_shapes='circle')
probe1.move([250, -90])

##############################################################################
# Create a ProbeBunch and
# add the Probe into it

probebunch = ProbeBunch()
probebunch.add_probe(probe0)
probebunch.add_probe(probe1)

print('probe0.get_electrode_count()', probe0.get_electrode_count())
print('probe1.get_electrode_count()', probe1.get_electrode_count())
print('probebunch.get_channel_count()', probebunch.get_channel_count())

##############################################################################
# Plot all probe in the same axes

plot_probe_bunch(probebunch, separate_axes=False)

##############################################################################
# Plot all probe in seperated axes

plot_probe_bunch(probebunch, separate_axes=True,  with_channel_index=True)

plt.show()





