"""
Import export to files
----------------------

probeinterface have its own format based on JSON.
The format handle several probe in one file.
It have 'probes' key that hanle a list of probe.

Each probe dict in the json folow the class attributes.

It also support read from theses formats: (and sometimes write)

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
# Generate 2 dummy Probe with util function
# and a ProbeGroup

probe0 = generate_dummy_probe(elec_shapes='square')
probe1 = generate_dummy_probe(elec_shapes='circle')
probe1.move([250, -90])

probegroup = ProbeGroup()
probegroup.add_probe(probe0)
probegroup.add_probe(probe1)

##############################################################################
# probe interface have its own format hdf5 based that store one ProbeGroup (and so several Probe)

write_probeinterface('my_two_probe_setup.json', probegroup)

probegroup2 = read_probeinterface('my_two_probe_setup.json')
plot_probe_group(probegroup2)


##############################################################################
# The format looks like this

with open('my_two_probe_setup.json', mode='r') as f:
    txt = f.read()

print(txt[:600], '...')


##############################################################################
# PRB is an historical format introduce by klusta team and also used by spikeinterface/spyeking-circus/tridesclous
# The format looks like this and is in fact a python script that describe a dict.
# This format handle:
#   * multi group (multi shank or multi probe)
#   * electrode_positions with 'geometry'
#   * device_channel_indeices with 'channels'
#
# Lets make a file with this and read back into ProbGroup.
#
# Here an example with 


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
plot_probe_group(two_tetrode, same_axe=False, with_channel_index=True)


plt.show()











