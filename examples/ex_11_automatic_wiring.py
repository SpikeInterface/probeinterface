"""
Automatic wiring
----------------

Here is an example on how to handle the wiring automatically and to get the device_channel_indices.
"""

##############################################################################
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe

##############################################################################
# Download one probe:

manufacturer = 'neuronexus'
probe_name = 'A1x32-Poly3-10mm-50-177'

probe = get_probe(manufacturer, probe_name)
print(probe)

##############################################################################
# We can "wire"  this probe to a recording device.
# Imagine we connect this Neuronexus probe with an Omnetic to an Intan RHD headstage.
#
# Using the wiring documentation from these two sites:
# https://www.neuronexus.com/files/wiringconfiguration/Wiring_H32.pdf
# http://intantech.com/RHD_headstages.html?tabSelect=RHD32ch&yPos=0
#
# After a long headache we can figure out the wiring to the device manually and set it
# using the `probe.set_device_channel_indices()` function:

device_channel_indices = [
    16, 17, 18, 20, 21, 22, 31, 30, 29, 27, 26, 25, 24, 28, 23, 19,
    12, 8, 3, 7, 6, 5, 4, 2, 1, 0, 9, 10, 11, 13, 14, 15]
probe.set_device_channel_indices(device_channel_indices)

##############################################################################
# In order to ease this process, `probeinterface` also includes some commonly
# used wirings based on standard connectors. We created references and spreadsheets
# showing how these wirings are computed at `probeinterface/resources/wiring_references
# <https://github.com/SpikeInterface/probeinterface/tree/9776684948e3ceba71601e5c0f90c2672f665234/resources>`_.
# If we have a Intan RHD2132 Headstage attached to a NeuroNexus H32 Connector, we can
# import this wiring as follows:

probe.wiring_to_device('H32>RHD2132')
print(probe.device_channel_indices)

##############################################################################
# In this figure we have 2 numbers for each contact:
#    * the upper number "prbXX" is the contact id (one-based, from NeuroNexus)
#    * the lower "devXX" is the channel on the Intan device (zero-based)

fig, ax = plt.subplots(figsize=(5, 15))
plot_probe(probe, with_contact_id=True, with_device_index=True, ax=ax)


##############################################################################
# Available wiring "pathways"
# ---------------------------
#
# The available pathways can be found in the `probeinterface.wiring <https://github.com/SpikeInterface/probeinterface/blob/main/src/probeinterface/wiring.py>`_ module.
#
# The following pathways are available:
#

from probeinterface import get_available_pathways

print(get_available_pathways())

plt.show()
