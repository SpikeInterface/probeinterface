"""
Multi shank probes.
-----------------------------------

Some probes have multi shank.
In probeinterface it is still a probe but internally
each probe handle an internale `shank_ids` vector.

Optionally, a :py:class:`Probe` object can render split into :py:class:`Shank`.

For spike sorting, very often each shank is computed separatly.

Here an example.





"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup
from probeinterface import generate_linear_probe, generate_multi_shank
from probeinterface import combinate_probes
from probeinterface.plotting import plot_probe



##############################################################################
# Let use a generator to create a multi shank


multi_shank = generate_multi_shank(num_shank=3, num_columns=2, num_elec_per_column=6)
plot_probe(multi_shank)


##############################################################################
# It is one probe but internally the `Probe.shank_ids` vector handle the shank.

print(multi_shank.shank_ids)

##############################################################################
#  we can iterate and get a :py:class:`Shank`
# A Shank is link to a Pobe object and can also retrieve posistions/electrode shape/...

for i, shank in enumerate(multi_shank.get_shanks()):
    print('shank', i)
    print(shank.__class__)
    print(shank.get_electrode_count())
    print(shank.electrode_positions.shape)


##############################################################################
# Another trick to create multi shank Probe is to create several Shank as separate Probe
# and then combinate then into a single Probe object

# generate a 2 shanks linear
probe0 = generate_linear_probe(num_elec=16,  ypitch=20,
                electrode_shapes='square',
                electrode_shape_params={'width': 12})
probe1 = probe0.copy()
probe1.move([100,0])

multi_shank = combinate_probes([probe0, probe1])

##############################################################################

print(multi_shank.shank_ids)


##############################################################################

plot_probe(multi_shank)

plt.show()





