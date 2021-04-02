"""
Generate a ProbeGroup
---------------------

This example shows how to assemble several Probe objects into a ProbeGroup object.

"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe_group
from probeinterface import generate_dummy_probe

##############################################################################
# Generate 2 dummy `Probe` objects with the utils function:
# 

probe0 = generate_dummy_probe(elec_shapes='square')
probe1 = generate_dummy_probe(elec_shapes='circle')
probe1.move([250, -90])

##############################################################################
# Let's create a `ProbeGroup` and
# add the `Probe` objects into it:

probegroup = ProbeGroup()
probegroup.add_probe(probe0)
probegroup.add_probe(probe1)

print('probe0.get_contact_count()', probe0.get_contact_count())
print('probe1.get_contact_count()', probe1.get_contact_count())
print('probegroup.get_channel_count()', probegroup.get_channel_count())

##############################################################################
#  We can now plot all probes in the same axis:

plot_probe_group(probegroup, same_axes=True)

##############################################################################
#  or in separate axes:

plot_probe_group(probegroup, same_axes=False, with_channel_index=True)

plt.show()
