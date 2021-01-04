"""
Get probe from library
---------------------------------------

Probe interface provide a libray of probes from several manufactrers on the gin platform
here https://gin.g-node.org/spikeinterface/probeinterface_library

User and manufacturer can so contribute to it.

The python module provide a function to download and cache files locally
with probeinterface format json based.
"""


##############################################################################
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe


##############################################################################
# Dowload one probe

manufacturer = 'neuronexus'
probe_name = 'A1x32-Poly3-10mm-50-177'

probe = get_probe(manufacturer, probe_name)
print(probe)


##############################################################################
# Files from library contain also so annotations specific to manufacturers
# We can see here that neuronexus handle electrode index starting at "1" (one based)

pprint(probe.annotations)

##############################################################################
# So when plotting the channel index are automatically displayed with one based
# even if internally everything is still zero based

plot_probe(probe, with_channel_index=True)


##############################################################################

plt.show()
