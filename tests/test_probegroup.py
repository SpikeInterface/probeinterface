from probeinterface import ProbeGroup
from probeinterface import generate_dummy_probe

import pytest

import numpy as np


def _make_probegroup():
    """Fixture: a ProbeGroup with 3 probes, each with device channel indices set."""
    probegroup = ProbeGroup()
    nchan = 0
    for i in range(3):
        probe = generate_dummy_probe()
        probe.move([i * 100, i * 80])
        n = probe.get_contact_count()
        probe.set_device_channel_indices(np.arange(n) + nchan)
        probegroup.add_probe(probe, probe_id=f"probe_00{i}")
        nchan += n
    return probegroup


@pytest.fixture
def probegroup():
    return _make_probegroup()


def test_probegroup(probegroup):
    indices = probegroup.get_global_device_channel_indices()

    ids = probegroup.get_global_contact_ids()

    df = probegroup.to_dataframe()

    arr = probegroup.to_numpy(complete=False)
    other = ProbeGroup.from_numpy(arr)
    arr = probegroup.to_numpy(complete=True)
    other = ProbeGroup.from_numpy(arr)

    d = probegroup.to_dict()
    other = ProbeGroup.from_dict(d)
    assert probegroup.probe_ids == other.probe_ids


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

    with pytest.raises(ValueError):
        probe.set_contact_ids(["a", "a"])


def test_set_contact_ids_rejects_wrong_size():
    """Setting contact_ids with wrong count raises ValueError."""
    from probeinterface import Probe

    positions = np.array([[0, 0], [10, 10]])
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})

    with pytest.raises(ValueError):
        probe.set_contact_ids(["a", "b", "c"])


# ── get_global_contact_positions() tests ────────────────────────────────────


def test_get_global_contact_positions_shape(probegroup):
    pos = probegroup.get_global_contact_positions()
    assert pos.shape == (probegroup.get_contact_count(), probegroup.ndim)


def test_get_global_contact_positions_matches_per_probe(probegroup):
    pos = probegroup.get_global_contact_positions()
    offset = 0
    for probe in probegroup.probes:
        n = probe.get_contact_count()
        np.testing.assert_array_equal(pos[offset : offset + n], probe.contact_positions)
        offset += n


def test_get_global_contact_positions_single_probe(probegroup):
    pos = probegroup.get_global_contact_positions()
    np.testing.assert_array_equal(
        pos[: probegroup.probes[0].get_contact_count()], probegroup.probes[0].contact_positions
    )


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


# ── copy() tests ────────────────────────────────────────────────────────────


def test_copy_returns_new_object(probegroup):
    pg_copy = probegroup.copy()
    assert pg_copy is not probegroup
    assert len(pg_copy.probes) == len(probegroup.probes)
    for orig, copied in zip(probegroup.probes, pg_copy.probes):
        assert orig is not copied


def test_copy_preserves_positions(probegroup):
    pg_copy = probegroup.copy()
    for orig, copied in zip(probegroup.probes, pg_copy.probes):
        np.testing.assert_array_equal(orig.contact_positions, copied.contact_positions)


def test_copy_preserves_device_channel_indices(probegroup):
    pg_copy = probegroup.copy()
    np.testing.assert_array_equal(
        probegroup.get_global_device_channel_indices(),
        pg_copy.get_global_device_channel_indices(),
    )


def test_copy_preserves_contact_ids(probegroup):
    """Probe.copy() preserves contact_ids when they are set on the probe."""
    for index, probe in enumerate(probegroup.probes):
        n = probe.get_contact_count()
        probe.set_contact_ids([f"p{index}-c{i}" for i in range(n)])

    pg_copy = probegroup.copy()

    original_ids = probegroup.get_global_contact_ids()
    copied_ids = pg_copy.get_global_contact_ids()
    np.testing.assert_array_equal(copied_ids, original_ids)


def test_copy_is_independent(probegroup):
    """Mutating the copy must not affect the original."""
    original_positions = probegroup.probes[0].contact_positions.copy()
    pg_copy = probegroup.copy()
    pg_copy.probes[0].move([999, 999])
    np.testing.assert_array_equal(probegroup.probes[0].contact_positions, original_positions)


# ── get_slice() simple : natural order


def test_get_slice_by_bool(probegroup):
    total = probegroup.get_contact_count()
    sel = np.zeros(total, dtype=bool)
    sel[:5] = True  # first 5 contacts from the first probe
    sliced = probegroup.get_slice(sel)
    assert sliced.get_contact_count() == 5


def test_get_slice_by_index(probegroup):
    indices = np.array([0, 1, 2, 33, 34])  # contacts from both probes
    sliced = probegroup.get_slice(indices)
    assert sliced.get_contact_count() == 5


def test_get_slice_preserves_device_channel_indices(probegroup):
    indices = np.array([0, 1, 2])
    sliced = probegroup.get_slice(indices)
    orig_chans = probegroup.get_global_device_channel_indices()["device_channel_indices"][:3]
    sliced_chans = sliced.get_global_device_channel_indices()["device_channel_indices"]
    np.testing.assert_array_equal(sliced_chans, orig_chans)


def test_get_slice_preserves_positions(probegroup):
    indices = np.array([0, 1, 2])
    sliced = probegroup.get_slice(indices)
    expected = probegroup.get_global_contact_positions()[indices]
    np.testing.assert_array_equal(sliced.get_global_contact_positions(), expected)


# def test_get_slice_empty_selection(probegroup):
#     sliced = probegroup.get_slice(np.array([], dtype=int))
#     assert sliced.get_contact_count() == 0
#     assert len(sliced.probes) == 0


def test_get_slice_wrong_bool_size(probegroup):
    with pytest.raises(AssertionError):
        probegroup.get_slice(np.array([True, False]))  # wrong size


def test_get_slice_out_of_bounds(probegroup):
    total = probegroup.get_contact_count()
    with pytest.raises(AssertionError):
        probegroup.get_slice(np.array([total + 10]))


def test_get_slice_all_contacts(probegroup):
    """Slicing with all contacts should give an equivalent ProbeGroup."""
    total = probegroup.get_contact_count()
    sliced = probegroup.get_slice(np.arange(total))
    assert sliced.get_contact_count() == total
    np.testing.assert_array_equal(
        sliced.get_global_contact_positions(),
        probegroup.get_global_contact_positions(),
    )


# ── get_slice : probe annotations and probe_ids propagation ─────────────────


def _annotated_probegroup():
    """ProbeGroup with 3 probes, each carrying distinct annotations and probe_id."""
    pg = ProbeGroup()
    for i in range(3):
        probe = generate_dummy_probe()
        probe.move([i * 200, 0])
        probe.annotate(brain_area=f"area_{i}", shank=f"s{i}")
        pg.add_probe(probe, probe_id=f"probe_{i}")
    return pg


def test_get_slice_propagates_annotations():
    """Annotations of each original probe are propagated to the sliced probe."""
    pg = _annotated_probegroup()
    n_each = pg.probes[0].get_contact_count()

    # take a few contacts from each of the 3 probes
    sel = np.array([0, 1, n_each, n_each + 1, 2 * n_each, 2 * n_each + 1])
    sub = pg.get_slice(sel)

    assert len(sub.probes) == 3
    for i, probe in enumerate(sub.probes):
        assert probe.annotations["brain_area"] == f"area_{i}"
        assert probe.annotations["shank"] == f"s{i}"


def test_get_slice_maps_annotations_to_correct_probe_when_skipping():
    """
    When the selection skips a middle probe, annotations must still map to the
    correct sliced probe (not shift by position).
    """
    pg = _annotated_probegroup()
    n_each = pg.probes[0].get_contact_count()

    # contacts only from probe 0 and probe 2 (probe 1 is skipped entirely)
    sel = np.zeros(pg.get_contact_count(), dtype=bool)
    sel[0:3] = True
    sel[2 * n_each : 2 * n_each + 4] = True
    sub = pg.get_slice(sel)

    assert len(sub.probes) == 2
    # first sliced probe corresponds to original probe 0, second to original probe 2
    assert sub.probes[0].annotations["brain_area"] == "area_0"
    assert sub.probes[1].annotations["brain_area"] == "area_2"
    assert sub.probes[0].get_contact_count() == 3
    assert sub.probes[1].get_contact_count() == 4


def test_get_slice_sets_probe_ids():
    """probe_ids are carried over to the sliced ProbeGroup."""
    pg = _annotated_probegroup()
    n_each = pg.probes[0].get_contact_count()

    sel = np.array([0, 1, n_each, 2 * n_each])
    sub = pg.get_slice(sel)
    assert sub.probe_ids == ["probe_0", "probe_1", "probe_2"]


def test_get_slice_sets_probe_ids_when_skipping():
    """probe_ids reflect only the probes present in the selection, in order."""
    pg = _annotated_probegroup()
    n_each = pg.probes[0].get_contact_count()

    # contacts only from probe 0 and probe 2
    sel = np.array([0, 2 * n_each])
    sub = pg.get_slice(sel)
    assert len(sub.probes) == 2
    assert sub.probe_ids == ["probe_0", "probe_2"]


def test_get_slice_single_probe_keeps_probe_id_and_annotations():
    """Slicing contacts from a single probe keeps that probe's id and annotations."""
    pg = _annotated_probegroup()
    n_each = pg.probes[0].get_contact_count()

    sel = np.arange(n_each, n_each + 3)  # only probe 1
    sub = pg.get_slice(sel)
    assert len(sub.probes) == 1
    assert sub.probe_ids == ["probe_1"]
    assert sub.probes[0].annotations["brain_area"] == "area_1"


# ── global_contact_order : to_numpy/from_numpy, to_dict/from_dict, get_slice


def test_reordered_probegroup(probegroup):
    order = np.concatenate([np.arange(0, 96, 2), np.arange(95, 0, -2)])

    contact_vector = probegroup.to_numpy(complete=True)
    contact_vector = contact_vector[order]

    probegroup2 = ProbeGroup.from_numpy(contact_vector)
    assert probegroup2._global_contact_order is not None
    contact_vector2 = probegroup2.to_numpy(complete=True)
    assert np.array_equal(contact_vector, contact_vector2)

    probegroup3 = ProbeGroup.from_dict(probegroup2.to_dict())
    assert probegroup3._global_contact_order is not None
    contact_vector3 = probegroup3.to_numpy(complete=True)
    assert np.array_equal(contact_vector2, contact_vector3)

    probegroup4 = probegroup.get_slice(order)
    assert probegroup4._global_contact_order is not None
    contact_vector4 = probegroup4.to_numpy(complete=True)
    assert np.array_equal(contact_vector3, contact_vector4)

    probegroup5 = ProbeGroup.from_dict(probegroup4.to_dict())
    assert probegroup5._global_contact_order is not None
    contact_vector5 = probegroup5.to_numpy(complete=True)
    assert np.array_equal(contact_vector4, contact_vector5)

    # let go back to original order
    rev_order = np.argsort(order)
    probegroup6 = probegroup5.get_slice(rev_order)
    assert probegroup6._global_contact_order is None


def _interleaved_order():
    """An order interleaving contacts across probes (non-natural)."""
    return np.concatenate([np.arange(0, 96, 2), np.arange(95, 0, -2)])


def test_global_contact_order_natural_is_none(probegroup):
    """A non-interleaved (natural) contact vector does not set a custom order."""
    pg = ProbeGroup.from_numpy(probegroup.to_numpy(complete=True))
    assert pg._global_contact_order is None


def test_global_contact_order_positions_reflect_order(probegroup):
    """get_global_contact_positions follows the custom global contact order."""
    order = _interleaved_order()
    natural_positions = probegroup.get_global_contact_positions().copy()

    pg = ProbeGroup.from_numpy(probegroup.to_numpy(complete=True)[order])
    assert pg._global_contact_order is not None
    np.testing.assert_array_equal(pg.get_global_contact_positions(), natural_positions[order])


def test_global_contact_order_ids_reflect_order(probegroup):
    """get_global_contact_ids follows the custom global contact order."""
    order = _interleaved_order()
    natural_ids = probegroup.get_global_contact_ids().copy()

    pg = ProbeGroup.from_numpy(probegroup.to_numpy(complete=True)[order])
    np.testing.assert_array_equal(pg.get_global_contact_ids(), natural_ids[order])


def test_global_contact_order_device_channel_indices_roundtrip(probegroup):
    """
    With a custom global contact order, device_channel_indices are zipped to the
    (reordered) to_numpy() vector. Setting them must roundtrip through both
    to_numpy() and get_global_device_channel_indices().
    """
    order = _interleaved_order()
    pg = ProbeGroup.from_numpy(probegroup.to_numpy(complete=True)[order])
    assert pg._global_contact_order is not None

    n = pg.get_contact_count()
    device_channel_indices = np.arange(n)
    pg.set_global_device_channel_indices(device_channel_indices)

    got = pg.to_numpy(complete=True)["device_channel_indices"]
    np.testing.assert_array_equal(got, device_channel_indices)

    got_getter = pg.get_global_device_channel_indices()["device_channel_indices"]
    np.testing.assert_array_equal(got_getter, device_channel_indices)


# ── select_contacts() tests ─────────────────────────────────────────────────


def _probegroup_with_contact_ids(unique=True):
    """ProbeGroup with 3 probes whose contact_ids are unique (or duplicated) across probes."""
    pg = ProbeGroup()
    for i in range(3):
        probe = generate_dummy_probe()
        probe.move([i * 100, i * 80])
        n = probe.get_contact_count()
        if unique:
            probe.set_contact_ids([f"p{i}c{j}" for j in range(n)])
        else:
            probe.set_contact_ids([f"c{j}" for j in range(n)])
        pg.add_probe(probe)
    return pg


def test_select_contacts_unique_ids():
    """Selecting by globally unique contact ids returns exactly those contacts."""
    pg = _probegroup_with_contact_ids(unique=True)
    selected_ids = ["p0c0", "p0c1", "p2c5"]
    sub = pg.select_contacts(selected_ids)

    assert sub.get_contact_count() == 3
    # contacts come from two distinct probes
    assert len(sub.probes) == 2
    assert set(sub.get_global_contact_ids()) == set(selected_ids)


def test_select_contacts_single_probe():
    """Selecting contacts from a single probe keeps a single probe."""
    pg = _probegroup_with_contact_ids(unique=True)
    sub = pg.select_contacts(["p1c0", "p1c1", "p1c2"])
    assert sub.get_contact_count() == 3
    assert len(sub.probes) == 1


def test_select_contacts_ambiguous_ids_without_probe_ids_raises():
    """
    Without probe_ids, a contact id that exists on more than one probe is
    ambiguous and raises a ValueError naming the offending id(s).
    """
    pg = _probegroup_with_contact_ids(unique=False)
    with pytest.raises(ValueError, match="c0"):
        pg.select_contacts(["c0"])


def test_select_contacts_with_probe_ids():
    """probe_ids (paired with contact_ids) disambiguate duplicated contact ids."""
    pg = _probegroup_with_contact_ids(unique=False)
    sub = pg.select_contacts(["c0", "c1"], probe_ids=["1", "1"])
    assert sub.get_contact_count() == 2
    assert len(sub.probes) == 1
    np.testing.assert_array_equal(sorted(sub.get_global_contact_ids()), ["c0", "c1"])


def test_select_contacts_same_id_across_probes_with_probe_ids():
    """The same contact id can be selected from several probes using probe_ids."""
    pg = _probegroup_with_contact_ids(unique=False)
    sub = pg.select_contacts(["c0", "c0"], probe_ids=["0", "2"])
    assert sub.get_contact_count() == 2
    assert len(sub.probes) == 2


def test_select_contacts_probe_ids_length_mismatch_raises():
    """probe_ids must have the same length as contact_ids."""
    pg = _probegroup_with_contact_ids(unique=False)
    with pytest.raises(ValueError):
        pg.select_contacts(["c0", "c1"], probe_ids=["0"])


def test_select_contacts_too_many_ids_without_probe_ids_raises():
    """
    Requesting more contact ids than the number of unique ids without probe_ids
    raises a ValueError.
    """
    pg = _probegroup_with_contact_ids(unique=False)
    n_unique = len(np.unique(pg.get_global_contact_ids()))
    too_many = [f"c{j}" for j in range(n_unique + 1)]
    with pytest.raises(ValueError):
        pg.select_contacts(too_many)


def test_select_contacts_follows_requested_order():
    """The selection follows the order of the provided contact_ids, even across probes."""
    pg = _probegroup_with_contact_ids(unique=True)
    # interleave contacts from different probes in a non-natural order
    selected_ids = ["p2c5", "p0c1", "p1c0", "p0c0"]
    sub = pg.select_contacts(selected_ids)

    np.testing.assert_array_equal(sub.get_global_contact_ids(), selected_ids)

    # positions must follow the same order as the requested ids
    all_ids = pg.get_global_contact_ids()
    all_positions = pg.get_global_contact_positions()
    expected = np.vstack([all_positions[all_ids == cid] for cid in selected_ids])
    np.testing.assert_array_equal(sub.get_global_contact_positions(), expected)


def test_select_probes_keeps_every_contact_of_matching_probes():
    """select_probes keeps every contact of the matching probes."""
    pg = _probegroup_with_contact_ids(unique=False)
    n_per_probe = pg.probes[0].get_contact_count()

    sub_str = pg.select_probes("1")
    assert sub_str.get_contact_count() == n_per_probe
    assert len(sub_str.probes) == 1

    sub_one = pg.select_probes(["1"])
    assert sub_one.get_contact_count() == n_per_probe
    assert len(sub_one.probes) == 1

    sub_two = pg.select_probes(["1", "2"])
    assert sub_two.get_contact_count() == 2 * n_per_probe
    assert len(sub_two.probes) == 2


def test_select_probes_keeps_array_order():
    """select_probes preserves the contact order."""
    pg = _probegroup_with_contact_ids(unique=False)
    sub = pg.select_probes(["2", "0"])
    # even if we requested probes in a different order, the contacts are still ordered by their original global order
    probe_index_per_contact = sub.to_numpy(complete=True)["probe_id"]
    assert probe_index_per_contact[0] == "0"


def test_select_probes_single_probe():
    """Selecting a single probe keeps a single probe with its contact ids."""
    pg = _probegroup_with_contact_ids(unique=True)
    sub = pg.select_probes(["1"])
    assert len(sub.probes) == 1
    assert sub.probe_ids == ["1"]
    assert all(cid.startswith("p1") for cid in sub.get_global_contact_ids())


def test_select_probes_preserves_probe_ids():
    """The selected ProbeGroup keeps the requested probe ids."""
    pg = _probegroup_with_contact_ids(unique=False)
    sub = pg.select_probes(["2", "0"])
    assert set(sub.probe_ids) == {"0", "2"}


def test_select_probes_preserves_positions():
    """Contacts of the selected probes keep their global positions."""
    pg = _probegroup_with_contact_ids(unique=True)

    all_ids = pg.get_global_contact_ids()
    all_positions = pg.get_global_contact_positions()

    sub = pg.select_probes(["0", "2"])
    sub_ids = sub.get_global_contact_ids()
    sub_positions = sub.get_global_contact_positions()
    for cid, pos in zip(sub_ids, sub_positions):
        np.testing.assert_array_equal(pos, all_positions[all_ids == cid][0])


def test_select_probes_none_raises():
    """Calling select_probes without probe_ids raises a ValueError."""
    pg = _probegroup_with_contact_ids(unique=False)
    with pytest.raises(ValueError):
        pg.select_probes(None)


def test_select_probes_all_probes():
    """Selecting all probes returns the whole ProbeGroup."""
    pg = _probegroup_with_contact_ids(unique=True)
    sub = pg.select_probes(["0", "1", "2"])
    assert sub.get_contact_count() == pg.get_contact_count()
    assert len(sub.probes) == len(pg.probes)


def test_select_contacts_duplicated_ids_raises():
    """Passing the same contact id more than once raises a ValueError."""
    pg = _probegroup_with_contact_ids(unique=True)
    with pytest.raises(ValueError):
        pg.select_contacts(["p0c0", "p0c1", "p0c0"])


def test_select_contacts_preserves_order_in_array():
    """Selected contacts keep the order specified in the input array."""
    pg = _probegroup_with_contact_ids(unique=True)
    contact_ids_list = [
        ["p0c1", "p0c0", "p2c5"],
        ["p2c5", "p0c0", "p0c1"],
        [
            "p0c1",
            "p2c5",
            "p0c0",
        ],
    ]
    for selected_ids in contact_ids_list:
        sub = pg.select_contacts(selected_ids)
        contact_vector = sub.to_numpy(complete=True)
        sub_ids = contact_vector["contact_ids"]
        assert list(sub_ids) == selected_ids


def test_select_contacts_preserves_positions():
    """Selected contacts keep their global positions."""
    pg = _probegroup_with_contact_ids(unique=True)
    selected_ids = ["p0c0", "p0c1", "p2c5"]

    all_ids = pg.get_global_contact_ids()
    all_positions = pg.get_global_contact_positions()
    expected = np.vstack([all_positions[all_ids == cid] for cid in selected_ids])

    sub = pg.select_contacts(selected_ids)
    sub_ids = sub.get_global_contact_ids()
    sub_positions = sub.get_global_contact_positions()
    got = np.vstack([sub_positions[sub_ids == cid] for cid in selected_ids])

    np.testing.assert_array_equal(got, expected)


# ── add_probe : default probe_id generation ─────────────────────────────────


def test_add_probe_default_id_does_not_recycle_after_gap():
    """
    The default probe_id must not collide with an existing id after a selection
    leaves a gap in the numeric ids. Using ``len(self._probes)`` would point back
    at an id that is still in use; ``max(numeric ids) + 1`` is gap-proof.
    """
    pg = ProbeGroup()
    for _ in range(3):
        pg.add_probe(generate_dummy_probe())
    assert pg.probe_ids == ["0", "1", "2"]

    # drop the middle probe -> ids become ["0", "2"], len is 2 (would collide with "2")
    sub = pg.select_probes(["0", "2"])
    assert sub.probe_ids == ["0", "2"]

    sub.add_probe(generate_dummy_probe())
    assert sub.probe_ids == ["0", "2", "3"]


def test_add_probe_default_id_with_non_numeric_ids():
    """
    With only non-numeric ids present, the generated id starts from "0" and can
    never collide with a non-numeric name.
    """
    pg = ProbeGroup()
    pg.add_probe(generate_dummy_probe(), probe_id="left")
    pg.add_probe(generate_dummy_probe(), probe_id="right")

    pg.add_probe(generate_dummy_probe())
    assert pg.probe_ids == ["left", "right", "0"]


def test_select_contacts_ambiguous_id_message_points_to_probe_ids():
    """
    When a contact id exists on several probes and no probe_ids are given, the
    error must guide the user to pass probe_ids rather than claim it cannot happen.
    """
    pg = _probegroup_with_contact_ids(unique=False)
    expected_error = """Some contact ids are ambiguous because they live on multiple probes; pass probe_ids to disambiguate which probe each belongs to:
"c0" lives on probes ['0', '1', '2']"""
    with pytest.raises(ValueError) as exc_info:
        pg.select_contacts(["c0"])
    assert str(exc_info.value) == expected_error


def test_select_contacts_reports_all_ambiguous_ids_at_once():
    """
    When several requested contact ids are ambiguous, the error lists all of them
    (with the probes each lives on) rather than failing on the first one.
    """
    pg = _probegroup_with_contact_ids(unique=False)
    expected_error = """Some contact ids are ambiguous because they live on multiple probes; pass probe_ids to disambiguate which probe each belongs to:
"c0" lives on probes ['0', '1', '2']
"c1" lives on probes ['0', '1', '2']"""
    with pytest.raises(ValueError) as exc_info:
        pg.select_contacts(["c0", "c1"])
    assert str(exc_info.value) == expected_error


def test_get_slice_preserves_planar_contour():
    """
    probe_planar_contour is a probe-level attribute (not part of the to_numpy
    dtype), so get_slice must copy it over explicitly instead of losing it.
    """
    pg = ProbeGroup()
    probe = generate_dummy_probe()
    contour = [[-10, -10], [-10, 100], [50, 120], [50, -10]]
    probe.set_planar_contour(contour)
    pg.add_probe(probe)

    sub = pg.get_slice(np.array([0, 1, 2]))
    assert sub.probes[0].probe_planar_contour is not None
    np.testing.assert_array_equal(sub.probes[0].probe_planar_contour, contour)


if __name__ == "__main__":
    probegroup = _make_probegroup()

    # test_probegroup(probegroup)
    # test_probegroup_3d()
    test_reordered_probegroup(probegroup)
