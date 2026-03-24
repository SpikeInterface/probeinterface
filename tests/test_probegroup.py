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


def _make_probegroup(n_probes=3):
    """Helper: build a ProbeGroup with device channel indices set."""
    probegroup = ProbeGroup()
    nchan = 0
    for i in range(n_probes):
        probe = generate_dummy_probe()
        probe.move([i * 100, i * 80])
        n = probe.get_contact_count()
        probe.set_device_channel_indices(np.arange(n) + nchan)
        probe.set_contact_ids([f"e{j}" for j in range(nchan, nchan + n)])
        nchan += n
    probegroup.add_probe(probe)
    return probegroup


def _make_probegroup_full(n_probes=3):
    """Helper: build a ProbeGroup where **every** probe is added."""
    probegroup = ProbeGroup()
    nchan = 0
    for i in range(n_probes):
        probe = generate_dummy_probe()
        probe.move([i * 100, i * 80])
        n = probe.get_contact_count()
        probe.set_device_channel_indices(np.arange(n) + nchan)
        probe.set_contact_ids([f"e{j}" for j in range(nchan, nchan + n)])
        probegroup.add_probe(probe)
        nchan += n
    return probegroup


# ── copy() tests ────────────────────────────────────────────────────────────


def test_copy_returns_new_object():
    pg = _make_probegroup_full(2)
    pg_copy = pg.copy()
    assert pg_copy is not pg
    assert len(pg_copy.probes) == len(pg.probes)
    for orig, copied in zip(pg.probes, pg_copy.probes):
        assert orig is not copied


def test_copy_preserves_positions():
    pg = _make_probegroup_full(2)
    pg_copy = pg.copy()
    for orig, copied in zip(pg.probes, pg_copy.probes):
        np.testing.assert_array_equal(orig.contact_positions, copied.contact_positions)


def test_copy_preserves_device_channel_indices():
    pg = _make_probegroup_full(2)
    pg_copy = pg.copy()
    np.testing.assert_array_equal(
        pg.get_global_device_channel_indices(),
        pg_copy.get_global_device_channel_indices(),
    )


def test_copy_does_not_preserve_contact_ids():
    """Probe.copy() intentionally does not copy contact_ids."""
    pg = _make_probegroup_full(2)
    pg_copy = pg.copy()
    # All contact_ids should be empty strings after copy
    assert all(cid == "" for cid in pg_copy.get_global_contact_ids())


def test_copy_is_independent():
    """Mutating the copy must not affect the original."""
    pg = _make_probegroup_full(2)
    original_positions = pg.probes[0].contact_positions.copy()
    pg_copy = pg.copy()
    pg_copy.probes[0].move([999, 999])
    np.testing.assert_array_equal(pg.probes[0].contact_positions, original_positions)


# ── get_slice() tests ───────────────────────────────────────────────────────


def test_get_slice_by_bool():
    pg = _make_probegroup_full(2)
    total = pg.get_contact_count()
    sel = np.zeros(total, dtype=bool)
    sel[:5] = True  # first 5 contacts from the first probe
    sliced = pg.get_slice(sel)
    assert sliced.get_contact_count() == 5


def test_get_slice_by_index():
    pg = _make_probegroup_full(2)
    indices = np.array([0, 1, 2, 33, 34])  # contacts from both probes
    sliced = pg.get_slice(indices)
    assert sliced.get_contact_count() == 5


def test_get_slice_preserves_device_channel_indices():
    pg = _make_probegroup_full(2)
    indices = np.array([0, 1, 2])
    sliced = pg.get_slice(indices)
    orig_chans = pg.get_global_device_channel_indices()["device_channel_indices"][:3]
    sliced_chans = sliced.get_global_device_channel_indices()["device_channel_indices"]
    np.testing.assert_array_equal(sliced_chans, orig_chans)


def test_get_slice_preserves_positions():
    pg = _make_probegroup_full(2)
    indices = np.array([0, 1, 2])
    sliced = pg.get_slice(indices)
    expected = pg.get_global_contact_positions()[indices]
    np.testing.assert_array_equal(sliced.get_global_contact_positions(), expected)


def test_get_slice_empty_selection():
    pg = _make_probegroup_full(2)
    sliced = pg.get_slice(np.array([], dtype=int))
    assert sliced.get_contact_count() == 0
    assert len(sliced.probes) == 0


def test_get_slice_wrong_bool_size():
    pg = _make_probegroup_full(2)
    with pytest.raises(AssertionError):
        pg.get_slice(np.array([True, False]))  # wrong size


def test_get_slice_out_of_bounds():
    pg = _make_probegroup_full(2)
    total = pg.get_contact_count()
    with pytest.raises(AssertionError):
        pg.get_slice(np.array([total + 10]))


def test_get_slice_all_contacts():
    """Slicing with all contacts should give an equivalent ProbeGroup."""
    pg = _make_probegroup_full(2)
    total = pg.get_contact_count()
    sliced = pg.get_slice(np.arange(total))
    assert sliced.get_contact_count() == total
    np.testing.assert_array_equal(
        sliced.get_global_contact_positions(),
        pg.get_global_contact_positions(),
    )


# ── get_global_contact_positions() tests ────────────────────────────────────


def test_get_global_contact_positions_shape():
    pg = _make_probegroup_full(3)
    pos = pg.get_global_contact_positions()
    assert pos.shape == (pg.get_contact_count(), pg.ndim)


def test_get_global_contact_positions_matches_per_probe():
    pg = _make_probegroup_full(3)
    pos = pg.get_global_contact_positions()
    offset = 0
    for probe in pg.probes:
        n = probe.get_contact_count()
        np.testing.assert_array_equal(pos[offset : offset + n], probe.contact_positions)
        offset += n


def test_get_global_contact_positions_single_probe():
    pg = _make_probegroup_full(1)
    pos = pg.get_global_contact_positions()
    np.testing.assert_array_equal(pos, pg.probes[0].contact_positions)


def test_get_global_contact_positions_3d():
    pg = ProbeGroup()
    for i in range(2):
        probe = generate_dummy_probe().to_3d()
        probe.move([i * 100, i * 80, i * 30])
        pg.add_probe(probe)
    pos = pg.get_global_contact_positions()
    assert pos.shape[1] == 3
    assert pos.shape[0] == pg.get_contact_count()


def test_get_global_contact_positions_reflects_move():
    """Positions should reflect probe movement."""
    pg = ProbeGroup()
    probe = generate_dummy_probe()
    original_pos = probe.contact_positions.copy()
    probe.move([50, 60])
    pg.add_probe(probe)
    pos = pg.get_global_contact_positions()
    np.testing.assert_array_equal(pos, original_pos + np.array([50, 60]))


if __name__ == "__main__":
    test_probegroup()
    # ~ test_probegroup_3d()
