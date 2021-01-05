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
# We can "wire"  this probe to device
# Imagine we connect this Neuronexus probe with Omenetic to an Intan RDH headstage
# 
# Using this 2 wiring documentation
# https://neuronexus.com/wp-content/uploads/2018/09/Wiring_H32.pdf
# http://intantech.com/RHD_headstages.html?tabSelect=RHD32ch&yPos=0
# 
# after long headhake we can do the wiring to device manually 
# using `probe.set_device_channel_indices(chan_indices)`

device_channel_indices = [
        16, 17, 18, 20, 21, 22, 31, 30, 29, 27, 26, 25, 24, 28, 23, 19, 
        12, 8, 3, 7, 6, 5, 4,  2, 1, 0, 9, 10, 11, 13, 14, 15]
probe.set_device_channel_indices(device_channel_indices)


##############################################################################
# but probeinterface also include internally some commonly used wiring based on classical pathway
# in our case it is

probe.wiring_to_device('H32>RDH')
print(probe.device_channel_indices)

##############################################################################
# In this figure we have 2 number for each contact:
#    * the upper number "prbXX" is the neuronexus  (indexing one based here for display)
#    * the lower "devXX" is the channel on device (intan)  (indexing zero based)

fig, ax = plt.subplots(figsize=(5, 15))
plot_probe(probe, with_channel_index=True, ax=ax)
ax.set_xlim(-100, 200)

plt.show()



