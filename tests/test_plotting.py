from probeinterface import Probe, ProbeGroup
from probeinterface import generate_dummy_probe, generate_dummy_probe_group
from probeinterface.plotting import plot_probe, plot_probegroup
from probeinterface.utils import get_auto_lims

import matplotlib.pyplot as plt
import numpy as np

import pytest


def test_plot_probe():
    probe = generate_dummy_probe()
    plot_probe(probe)
    plot_probe(probe, with_contact_id=True)
    plot_probe(probe, with_device_index=True)
    plot_probe(probe, text_on_contact=["abcde"[i % 5] for i in range(probe.get_contact_count())])

    # with color
    n = probe.get_contact_count()
    contacts_colors = np.random.rand(n, 3)
    plot_probe(probe, contacts_colors=contacts_colors)

    # 3d
    probe_3d = probe.to_3d(axes="xz")
    plot_probe(probe_3d)

    # on click
    probe.set_device_channel_indices(np.arange(probe.get_contact_count())[::-1])
    plot_probe(probe, show_channel_on_click=True)


def test_plot_probegroup():
    probegroup = generate_dummy_probe_group()

    plot_probegroup(probegroup, same_axes=True, with_contact_id=True)
    plot_probegroup(probegroup, same_axes=False)

    # 3d
    probegroup_3d = ProbeGroup()
    for probe in probegroup.probes:
        probegroup_3d.add_probe(probe.to_3d())
    probegroup_3d.probes[-1].move([0, 150, -50])
    plot_probegroup(probegroup_3d, same_axes=True)


def test_plot_probe_partial_lims():
    """Passing only one of xlims/ylims/zlims must not discard the ones that were given."""
    probe = generate_dummy_probe()
    auto_xlims, auto_ylims, _ = get_auto_lims(probe)

    # x only: xlims is honoured, ylims falls back to auto
    _, ax = plt.subplots()
    plot_probe(probe, ax=ax, xlims=(-11, 22))
    assert ax.get_xlim() == (-11, 22)
    assert ax.get_ylim() == pytest.approx(auto_ylims)

    # y only: ylims is honoured, xlims falls back to auto
    _, ax = plt.subplots()
    plot_probe(probe, ax=ax, ylims=(-33, 44))
    assert ax.get_ylim() == (-33, 44)
    assert ax.get_xlim() == pytest.approx(auto_xlims)

    # both given: neither is touched
    _, ax = plt.subplots()
    plot_probe(probe, ax=ax, xlims=(-11, 22), ylims=(-33, 44))
    assert ax.get_xlim() == (-11, 22)
    assert ax.get_ylim() == (-33, 44)

    # neither given: both are auto
    _, ax = plt.subplots()
    plot_probe(probe, ax=ax)
    assert ax.get_xlim() == pytest.approx(auto_xlims)
    assert ax.get_ylim() == pytest.approx(auto_ylims)


def test_plot_probe_partial_lims_3d():
    """In 3D, omitting zlims must not discard a supplied xlims or ylims."""
    probe_3d = generate_dummy_probe().to_3d(axes="xz")
    auto_xlims, auto_ylims, auto_zlims = get_auto_lims(probe_3d)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_probe(probe_3d, ax=ax, xlims=(-11, 22), ylims=(-33, 44))
    assert ax.get_xlim() == (-11, 22)
    assert ax.get_ylim() == (-33, 44)
    assert ax.get_zlim() == pytest.approx(auto_zlims)

    # zlims only: x and y fall back to auto
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_probe(probe_3d, ax=ax, zlims=(-55, 66))
    assert ax.get_zlim() == (-55, 66)
    assert ax.get_xlim() == pytest.approx(auto_xlims)
    assert ax.get_ylim() == pytest.approx(auto_ylims)


def test_plot_probe_two_side():
    probe = Probe()
    probe.set_contacts(
        positions=np.array(
            [
                [0, 0],
                [0, 10],
                [0, 20],
                [0, 0],
                [0, 10],
                [0, 20],
            ]
        ),
        shapes="circle",
        contact_ids=["F1", "F2", "F3", "B1", "B2", "B3"],
        contact_sides=["front", "front", "front", "back", "back", "back"],
    )

    plot_probe(probe, with_contact_id=True, side="front")
    plot_probe(probe, with_contact_id=True, side="back")


if __name__ == "__main__":
    # test_plot_probe()
    test_plot_probe_two_side()
    plt.show()
