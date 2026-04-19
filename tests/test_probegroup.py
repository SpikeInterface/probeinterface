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


def test_set_contact_ids_rejects_within_probe_duplicates():
    """Setting duplicate contact_ids within a single probe raises ValueError."""
    from probeinterface import Probe

    positions = np.array([[0, 0], [10, 10]])
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})

    with pytest.raises(ValueError, match="unique within a Probe"):
        probe.set_contact_ids(["a", "a"])


def test_set_contact_ids_rejects_wrong_size():
    """Setting contact_ids with wrong count raises ValueError."""
    from probeinterface import Probe

    positions = np.array([[0, 0], [10, 10]])
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})

    with pytest.raises(ValueError, match="do not have the same size"):
        probe.set_contact_ids(["a", "b", "c"])


def test_contact_vector_orders_connected_contacts():
    from probeinterface import Probe

    probe0 = Probe(ndim=2, si_units="um")
    probe0.set_contacts(
        positions=np.array([[10.0, 0.0], [30.0, 0.0]]),
        shapes="circle",
        shape_params={"radius": 5},
        shank_ids=["s0", "s1"],
        contact_sides=["front", "back"],
    )
    probe0.set_device_channel_indices([2, -1])

    probe1 = Probe(ndim=2, si_units="um")
    probe1.set_contacts(
        positions=np.array([[0.0, 0.0], [20.0, 0.0]]),
        shapes="circle",
        shape_params={"radius": 5},
        shank_ids=["s0", "s0"],
        contact_sides=["front", "front"],
    )
    probe1.set_device_channel_indices([0, 1])

    probegroup = ProbeGroup()
    probegroup.add_probe(probe0)
    probegroup.add_probe(probe1)

    probegroup._build_contact_vector()
    arr = probegroup._contact_vector

    assert arr.dtype.names == ("probe_index", "x", "y", "shank_ids", "contact_sides")
    assert arr.size == 3
    assert arr.flags.writeable is False
    assert np.array_equal(arr["probe_index"], np.array([1, 1, 0]))
    assert np.array_equal(arr["x"], np.array([0.0, 20.0, 10.0]))
    assert np.array_equal(np.column_stack((arr["x"], arr["y"])), np.array([[0.0, 0.0], [20.0, 0.0], [10.0, 0.0]]))


def test_contact_vector_cache_refresh_is_explicit():
    probegroup = ProbeGroup()
    probe = generate_dummy_probe()
    probe.set_device_channel_indices(np.arange(probe.get_contact_count()))
    probegroup.add_probe(probe)

    probegroup._build_contact_vector()
    dense_before = probegroup._contact_vector
    dense_before_again = probegroup._contact_vector
    assert dense_before is dense_before_again

    original_positions = np.column_stack((dense_before["x"], dense_before["y"])).copy()
    probe.move([5.0, 0.0])

    dense_after_move = probegroup._contact_vector
    assert dense_after_move is dense_before
    assert np.array_equal(np.column_stack((dense_after_move["x"], dense_after_move["y"])), original_positions)

    probegroup._build_contact_vector()
    dense_after_refresh = probegroup._contact_vector
    assert dense_after_refresh is not dense_before
    assert np.array_equal(
        np.column_stack((dense_after_refresh["x"], dense_after_refresh["y"])),
        original_positions + np.array([5.0, 0.0]),
    )

    probe.set_shank_ids(np.array(["a"] * probe.get_contact_count()))
    probegroup._build_contact_vector()
    dense_with_shanks = probegroup._contact_vector
    assert "shank_ids" in dense_with_shanks.dtype.names


def test_contact_vector_requires_wired_contacts():
    probegroup = ProbeGroup()
    probe = generate_dummy_probe()
    probegroup.add_probe(probe)

    with pytest.raises(ValueError, match="requires at least one wired contact"):
        probegroup._build_contact_vector()


def test_contact_vector_supports_3d_positions():
    probegroup = ProbeGroup()
    probe = generate_dummy_probe().to_3d()
    probe.set_device_channel_indices(np.arange(probe.get_contact_count()))
    probegroup.add_probe(probe)

    probegroup._build_contact_vector()
    dense = probegroup._contact_vector
    assert dense.dtype.names[:4] == ("probe_index", "x", "y", "z")
    assert np.column_stack((dense["x"], dense["y"], dense["z"])).shape[1] == 3


if __name__ == "__main__":
    test_probegroup()
    # ~ test_probegroup_3d()
