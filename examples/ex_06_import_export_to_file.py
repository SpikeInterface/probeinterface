"""
Import/export functions
-----------------------

`probeinterface` has its own format based on JSON.
The format can handle several probes in one file.
It has a 'probes' key that can contain a list of probes.

Each probe field in the json format contains the `Probe` class attributes.

It also supports reading (and sometimes writing) from theses formats:

  * PRB (.prb) : used by klusta/spyking-circus/tridesclous
  * CVS (.csv): 2 or 3 columns locations in text file
  * mearec (.h5) : mearec handle the geometry 
  * spikeglx (.meta) : spikeglx handle the handle also the geometry

"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probe_group
from probeinterface import generate_dummy_probe
from probeinterface import write_probeinterface, read_probeinterface
from probeinterface import write_prb, read_prb

##############################################################################
# Let's first generate 2 dummy probes and combine them
# into a ProbeGroup

probe0 = generate_dummy_probe(elec_shapes='square')
probe1 = generate_dummy_probe(elec_shapes='circle')
probe1.move([250, -90])

probegroup = ProbeGroup()
probegroup.add_probe(probe0)
probegroup.add_probe(probe1)

##############################################################################
# With the `write_probeinterface` and `read_probeinterface` functions we can
# write to and read from the json-based probeinterface format:

write_probeinterface('my_two_probe_setup.json', probegroup)

probegroup2 = read_probeinterface('my_two_probe_setup.json')
plot_probe_group(probegroup2)

##############################################################################
# The format looks like this:

with open('my_two_probe_setup.json', mode='r') as f:
    txt = f.read()

print(txt[:600], '...')

##############################################################################
# PRB is an historical format introduced by the Klusta team and it is also
# used by SpikeInterface, Spyking-circus, and Tridesclous.
# The format is in fact a python script that describes a dictionary.
# This format handles:
#   * multiple groups (multi-shank or multi-probe)
#   * contact_positions with 'geometry'
#   * device_channel_indices with 'channels'
#
# Here an example .prb file with 2 channel groups of 4 channels each.
# It can be easily loaded and plotted with `probeinterface`


prb_two_tetrodes = """
channel_groups = {
    0: {
            'channels' : [0,1,2,3],
            'geometry': {
                0: [0, 50],
                1: [50, 0],
                2: [0, -50],
                3: [-50, 0],
            }
    },
    1: {
            'channels' : [4,5,6,7],
            'geometry': {
                4: [0, 50],
                5: [50, 0],
                6: [0, -50],
                7: [-50, 0],
            }
    }
}
"""

with open('two_tetrodes.prb', 'w') as f:
    f.write(prb_two_tetrodes)

two_tetrode = read_prb('two_tetrodes.prb')
plot_probe_group(two_tetrode, same_axes=False, with_channel_index=True)

plt.show()
