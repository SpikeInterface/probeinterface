"""
Use to generate figure and format for documentation
"""

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup, combine_probes, write_probeinterface
from probeinterface.plotting import plot_probe, plot_probe_group

from probeinterface import generate_tetrode


probe0 = generate_tetrode(r=25)
probe0.create_auto_shape(probe_type='tip')
probe1 = generate_tetrode(r=25)
probe1.create_auto_shape(probe_type='tip')
probe1.move([150, 0])

probe = combine_probes([probe0, probe1])

pos = probe.contact_positions
pos[np.abs(pos)<0.0001] = 0
probe.contact_positions = pos

# do not include wiring in example : too complicated
#  probe.set_device_channel_indices([3,2,1,0, 7, 6, 5, 4])

probe.annotate(name='2 shank tetrodes', manufacturer='homemade')

print(probe.shank_ids)
d = probe.to_dict()
print(d.keys())

fig, ax = plt.subplots(figsize=(8, 8))
plot_probe(probe, with_channel_index=True, ax=ax)
ax.set_xlim(-50, 200)
ax.set_ylim(-150, 120)

write_probeinterface('probe_format_example.json', probe)


fig.savefig('img/probe_format_example.png')

plt.show()
