"""
Probe generator
--------------------------

probeinterface have also basic function to generate simple electrode layout like:

  * tetrode
  * linear probe
  * multi column probe

"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeBunch
from probeinterface.plotting import plot_probe, plot_probe_bunch
from probeinterface import generate_fake_probe

