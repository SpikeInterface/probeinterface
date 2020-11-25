from probeinterface import Probe, ProbeBunch
from probeinterface import generate_fake_probe, generate_fake_probe_bunch
from probeinterface.plotting import plot_probe, plot_probe_bunch

import matplotlib.pyplot as plt
import numpy as np

import pytest


def test_plot_probe():
    probe = generate_fake_probe()
    plot_probe(probe)
    plot_probe(probe, with_channel_index=True)
    
    # with color
    n = probe.get_electrode_count()
    electrode_colors = np.random.rand(n, 3)
    plot_probe(probe, electrode_colors = electrode_colors)
    
    # 3d
    probe_3d = probe.to_3d(plane='xz')
    plot_probe(probe_3d)


def test_plot_probe_bunch():
    probebunch  = generate_fake_probe_bunch()

    plot_probe_bunch(probebunch, separate_axes=False, with_channel_index=True)
    plot_probe_bunch(probebunch, separate_axes=True)
    
    # 3d
    probebunch_3d = ProbeBunch()
    for probe in probebunch.probes:
        probebunch_3d.add_probe(probe.to_3d())
    print(probebunch_3d.ndim)
    plot_probe_bunch(probebunch_3d, separate_axes=False)
    


if __name__ == '__main__':
    test_plot_probe()
    test_plot_probe_bunch()
    plt.show()
