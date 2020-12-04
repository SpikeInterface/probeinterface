"""
Handle channel indices
-----------------------------------

Probe have a complex electrodes index system due to layout.
When they are plug to a device like openephys with intan headstage, 
the channel order is mixed again. So the physical electrodes channel index
is rarely the channel index on the device.

This is why Prbe handle a separate `device_channel_indices`
"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeBunch
from probeinterface.plotting import plot_probe, plot_probe_bunch
from probeinterface import generate_multi_columns_probe

##############################################################################
# generate  probe
# note that here the wiring is not so complicated : each column increment the electrode index 
# from bottom to up

probe = generate_multi_columns_probe(num_columns=3,
                num_elec_per_column=[5, 6, 5],
                xpitch=75, ypitch=75, y_shift_per_column=[0, -37.5, 0],
                electrode_shapes='circle', electrode_shape_params={'radius': 12})


plot_probe(probe, with_channel_index=True)


##############################################################################
# The Probe is not connected to device yet

print(probe.device_channel_indices)

##############################################################################
# lets imagine a not very complicated headstage that mixed channel
# half of channel have natural indices other half is reserved

channel_indices = np.arange(16)
channel_indices[8:16] = channel_indices[8:16][::-1]
probe.set_device_channel_indices(channel_indices)
print(probe.device_channel_indices)

##############################################################################
# We can visualize the 2 indices: 
# 
#  * the prbXX is the electrode index ordered from 0 to N
#  * the devXX is the channel index on the device in another order

plot_probe(probe, with_channel_index=True)

##############################################################################
# Very often we have several probes on the device this lead to even complex channel indices
# `ProbeBunch.get_global_device_channel_indices()` give the overview of the device wiring.

#~ probe0 = generate_dummy_probe(elec_shapes='circle')
#~ probe1 = generate_dummy_probe(elec_shapes='square')
probe0 = generate_multi_columns_probe(num_columns=3,
                num_elec_per_column=[5, 6, 5],
                xpitch=75, ypitch=75, y_shift_per_column=[0, -37.5, 0],
                electrode_shapes='circle', electrode_shape_params={'radius': 12})
probe1 = probe0.copy()

probe1.move([350, 200])
probebunch = ProbeBunch()
probebunch.add_probe(probe0)
probebunch.add_probe(probe1)

# wire probe0 0 to 31
#  and shuffle
channel_indices0 = np.arange(16)
np.random.shuffle(channel_indices0)
probe0.set_device_channel_indices(channel_indices0)

# wire probe0 32 to 63
#  and shuffle
channel_indices1 = np.arange(16, 32)
np.random.shuffle(channel_indices1)
probe1.set_device_channel_indices(channel_indices1)

print(probebunch.get_global_device_channel_indices())

##############################################################################
# Can be also ploted
fig, ax = plt.subplots()
plot_probe_bunch(probebunch, with_channel_index=True, same_axe=True, ax=ax)
ax.set_xlim(-100, 600)
ax.set_ylim(-100, 600)


plt.show()

