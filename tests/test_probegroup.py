from probeinterface import ProbeGroup
from probeinterface import generate_dummy_probe

import pytest

import numpy as np


def test_probegroup():
    probegroup = ProbeGroup()

    nchan = 0
    for i in range(3):
        probe = generate_dummy_probe()
        probe.move([i * 100, i * 80])
        n = probe.get_contact_count()
        probe.set_device_channel_indices(np.arange(n)[::-1] + nchan)
        shank_ids = np.ones(n)
        shank_ids[: n // 2] *= i * 2
        shank_ids[n // 2 :] *= i * 2 + 1
        probe.set_shank_ids(shank_ids)
        probegroup.add_probe(probe)

        nchan += n

    indices = probegroup.get_global_device_channel_indices()

    ids = probegroup.get_global_contact_ids()

    df = probegroup.to_dataframe()
    # ~ print(df['global_contact_ids'])

    arr = probegroup.to_numpy(complete=False)
    other = ProbeGroup.from_numpy(arr)
    arr = probegroup.to_numpy(complete=True)
    other = ProbeGroup.from_numpy(arr)

    d = probegroup.to_dict()
    other = ProbeGroup.from_dict(d)

    # ~ from probeinterface.plotting import plot_probe_group, plot_probe
    # ~ import matplotlib.pyplot as plt
    # ~ plot_probe_group(probegroup)
    # ~ plot_probe_group(other)
    # ~ plt.show()

    # checking automatic generation of ids with new dummy probes
    probegroup.probes = []
    for i in range(3):
        probegroup.add_probe(generate_dummy_probe())
    probegroup.auto_generate_contact_ids()
    probegroup.auto_generate_probe_ids()

    for p in probegroup.probes:
        assert p.contact_ids is not None
        assert "probe_id" in p.annotations


def test_probegroup_3d():
    probegroup = ProbeGroup()

    for i in range(3):
        probe = generate_dummy_probe().to_3d()
        probe.move([i * 100, i * 80, i * 30])
        probegroup.add_probe(probe)

    assert probegroup.ndim == 3


def test_probegroup_allows_duplicate_positions_across_probes():
    """Test that ProbeGroup allows duplicate contact positions if they are in different probes."""
    from probeinterface import ProbeGroup, Probe
    import numpy as np

    # Probes have the same internal relative positions
    positions = np.array([[0, 0], [10, 10]])
    probe1 = Probe(ndim=2, si_units="um")
    probe1.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe2 = Probe(ndim=2, si_units="um")
    probe2.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})

    group = ProbeGroup()
    group.add_probe(probe1)
    group.add_probe(probe2)

    # Should not raise any error
    all_positions = np.vstack([p.contact_positions for p in group.probes])
    # There are duplicates across probes, but this is allowed
    assert (all_positions == [0, 0]).any()
    assert (all_positions == [10, 10]).any()
    # The group should have both probes
    assert len(group.probes) == 2


if __name__ == "__main__":
    test_probegroup()
    # ~ test_probegroup_3d()
