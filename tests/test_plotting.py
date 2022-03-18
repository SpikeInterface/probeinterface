from probeinterface import Probe, ProbeGroup
from probeinterface import generate_dummy_probe, generate_dummy_probe_group
from probeinterface.plotting import plot_probe, plot_probe_group

import matplotlib.pyplot as plt
import numpy as np

import pytest


def test_plot_probe():
    probe = generate_dummy_probe()
    plot_probe(probe)
    plot_probe(probe, with_channel_index=True)
    plot_probe(probe, with_contact_id=True)
    plot_probe(probe, with_device_index=True)
    plot_probe(probe, text_on_contact=['abcde'[i%5] for i in range(probe.get_contact_count())])

    # with color
    n = probe.get_contact_count()
    contacts_colors = np.random.rand(n, 3)
    plot_probe(probe, contacts_colors=contacts_colors)

    # 3d
    probe_3d = probe.to_3d(axes='xz')
    plot_probe(probe_3d)

    # on click
    probe.set_device_channel_indices(np.arange(probe.get_contact_count())[::-1])
    plot_probe(probe, show_channel_on_click=True)


def test_plot_probe_group():
    probegroup = generate_dummy_probe_group()

    plot_probe_group(probegroup, same_axes=True, with_channel_index=True)
    plot_probe_group(probegroup, same_axes=False)

    # 3d
    probegroup_3d = ProbeGroup()
    for probe in probegroup.probes:
        probegroup_3d.add_probe(probe.to_3d())
    probegroup_3d.probes[-1].move([0, 150, -50])
    plot_probe_group(probegroup_3d, same_axes=True)


if __name__ == '__main__':
    test_plot_probe()
    # test_plot_probe_group()
    plt.show()
