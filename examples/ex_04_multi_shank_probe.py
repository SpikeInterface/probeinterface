"""
Multi shank probes
------------------

This example shows how to deal with multi-shank probes.

In `probeinterface` this can be done with a `Probe` object, but internally
each probe handles a `shank_ids` vector to carry information about which contacts belong to which shanks.

Optionally, a :py:class:`Probe` object can be rendered split into :py:class:`Shank`.
"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup
from probeinterface import generate_linear_probe, generate_multi_shank
from probeinterface import combine_probes
from probeinterface.plotting import plot_probe

##############################################################################
# Let's use a generator to create a multi shank probe:


multi_shank = generate_multi_shank(num_shank=3, num_columns=2, num_contact_per_column=6)
plot_probe(multi_shank)

##############################################################################
# `multi_shank` is one `probe` object, but internally the `Probe.shank_ids` vector handles the shank ids.

print(multi_shank.shank_ids)

##############################################################################
# The dataframe displays the `shank_ids` column:

df = multi_shank.to_dataframe()
df

##############################################################################
#  We can iterate over a multi-shank probe and get :py:class:`Shank` objects.
# A `Shank` is link to a `Probe` object and can also retrieve
# positions, contact shapes, etc.:

for i, shank in enumerate(multi_shank.get_shanks()):
    print('shank', i)
    print(shank.__class__)
    print(shank.get_contact_count())
    print(shank.contact_positions.shape)

##############################################################################
# Another option to create multi-shank probes is to create several `Shank`
# objects as separate probes and then combine then into a single `Probe` object

# generate a 2 shanks linear
probe0 = generate_linear_probe(num_elec=16, ypitch=20,
                               contact_shapes='square',
                               contact_shape_params={'width': 12})
probe1 = probe0.copy()
probe1.move([100, 0])

multi_shank = combine_probes([probe0, probe1])

##############################################################################

print(multi_shank.shank_ids)

##############################################################################

plot_probe(multi_shank)

plt.show()
