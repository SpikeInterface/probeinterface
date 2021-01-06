"""
Generate a ProbeGroup
-----------------------------------

This code show how to assmble several Probe into a ProbeGroup object.

"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe_group
from probeinterface import generate_dummy_probe


##############################################################################
# Generate 2 dummy Probe with util function
# 

probe0 = generate_dummy_probe(elec_shapes='square')
probe1 = generate_dummy_probe(elec_shapes='circle')
probe1.move([250, -90])

##############################################################################
# Create a ProbeGroup and
# add the Probe into it

probegroup = ProbeGroup()
probegroup.add_probe(probe0)
probegroup.add_probe(probe1)

print('probe0.get_electrode_count()', probe0.get_electrode_count())
print('probe1.get_electrode_count()', probe1.get_electrode_count())
print('probegroup.get_channel_count()', probegroup.get_channel_count())

##############################################################################
# Plot all probe in the same axes

plot_probe_group(probegroup, same_axe=True)

##############################################################################
# Plot all probe in seperated axes

plot_probe_group(probegroup, same_axe=False,  with_channel_index=True)

plt.show()





