"""
Get probe from library
----------------------

`probeinterface` provides a library of probes from several manufacturers on the gin platform:
https://gin.g-node.org/spikeinterface/probeinterface_library

Users and manufacturers are welcome to contribute to it.

The Python module provide a function to download and cache files locally
in the `probeinterface` json-based format.
"""

##############################################################################
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe

##############################################################################
#  Download one probe:

manufacturer = 'neuronexus'
probe_name = 'A1x32-Poly3-10mm-50-177'

probe = get_probe(manufacturer, probe_name)
print(probe)

##############################################################################
# Files from the library also contain annotations specific to manufacturers.
# We can see here that Neuronexus probes have contact indices starting at "1" (one-based)

pprint(probe.annotations)

##############################################################################
# When plotting, the channel indices are automatically displayed with
# one-based notation (even if internally everything is still zero based):

plot_probe(probe, with_channel_index=True)

##############################################################################

plt.show()
