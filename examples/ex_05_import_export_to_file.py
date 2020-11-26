"""
Import export to files
-----------------------------------

probeinterface have its own format based on hdf5.
It is a trivial map between array and hfd tree.

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

from probeinterface import Probe, ProbeBunch
from probeinterface.plotting import plot_probe, plot_probe_bunch
from probeinterface import generate_fake_probe


