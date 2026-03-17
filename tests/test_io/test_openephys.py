import re
from pathlib import Path

import numpy as np

import pytest

import json

from probeinterface import read_openephys
from probeinterface.neuropixels_tools import _parse_openephys_settings, _select_openephys_probe_info
from probeinterface.neuropixels_tools import _slice_catalogue_probe, build_neuropixels_probe
from probeinterface.testing import validate_probe_dict

data_path = Path(__file__).absolute().parent.parent / "data" / "openephys"


def test_NP2_OE_1_0():
    # NP2 1-shank
    probeA = read_openephys(data_path / "OE_1.0_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeA")
    probe_dict = probeA.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probeA.get_shank_count() == 1
    assert probeA.get_contact_count() == 384


def test_NP2():
    # NP2
    probe = read_openephys(data_path / "OE_Neuropix-PXI" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 1


def test_NP2_four_shank():
    # NP2
    probe = read_openephys(data_path / "OE_Neuropix-PXI-NP2-4shank" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    # on this case, only shanks 2-3 are used
    assert probe.get_shank_count() == 2


def test_NP_Ultra():
    # ProbeD (NP1121) matches its catalogue geometry
    probeD = read_openephys(
        data_path / "OE_Neuropix-PXI-NP-Ultra" / "settings.xml",
        probe_name="ProbeD",
    )
    probe_dict = probeD.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probeD.get_shank_count() == 1
    assert probeD.get_contact_count() == 384
    # for this probe model, all channels are aligned
    assert len(np.unique(probeD.contact_positions[:, 0])) == 1


def test_probe_part_number_mismatch_with_catalogue():
    # ProbeA is labeled NP1100 but its positions don't match the NP1100 catalogue.
    # See https://github.com/SpikeInterface/probeinterface/issues/407
    expected_error = (
        "Could not match electrode positions to catalogue probe 'NP1100'. "
        "The probe part number in settings.xml may be incorrect. "
        "See https://github.com/SpikeInterface/probeinterface/issues/407 for details."
    )
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        read_openephys(
            data_path / "OE_Neuropix-PXI-NP-Ultra" / "settings.xml",
            probe_name="ProbeA",
        )


def test_NP1_subset():
    # NP1 - 200 channels selected by recording_state in Record Node
    probe_ap = read_openephys(data_path / "OE_Neuropix-PXI-subset" / "settings.xml", stream_name="ProbeA-AP")
    probe_dict = probe_ap.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probe_ap.get_shank_count() == 1
    assert "1.0" in probe_ap.description
    assert probe_ap.get_contact_count() == 200

    probe_lf = read_openephys(data_path / "OE_Neuropix-PXI-subset" / "settings.xml", stream_name="ProbeA-LFP")
    probe_dict = probe_lf.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probe_lf.get_shank_count() == 1
    assert "1.0" in probe_lf.description
    assert len(probe_lf.contact_positions) == 200

    # Not specifying the stream_name should raise an Exception, because both the ProbeA-AP and
    # ProbeA-LFP have custom channel selections
    with pytest.raises(AssertionError):
        probe = read_openephys(data_path / "OE_Neuropix-PXI-subset" / "settings.xml")


def test_multiple_probes():
    # multiple probes
    probeA = read_openephys(data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeA")
    probe_dict = probeA.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probeA.get_shank_count() == 1
    assert "1.0" in probeA.description

    probeB = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
        stream_name="RecordNode#ProbeB",
    )
    probe_dict = probeB.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probeB.get_shank_count() == 1

    probeC = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
        serial_number="20403311714",
    )
    probe_dict = probeC.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probeC.get_shank_count() == 1

    probeD = read_openephys(data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeD")
    probe_dict = probeD.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probeD.get_shank_count() == 1

    assert probeA.serial_number == "17131307831"
    assert probeB.serial_number == "20403311724"
    assert probeC.serial_number == "20403311714"
    assert probeD.serial_number == "21144108671"

    probeA2 = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings_2.xml",
        probe_name="ProbeA",
    )

    assert probeA2.get_shank_count() == 1
    ypos = probeA2.contact_positions[:, 1]
    assert np.min(ypos) >= 0

    probeB2 = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings_2.xml",
        probe_name="ProbeB",
    )

    assert probeB2.get_shank_count() == 1

    ypos = probeB2.contact_positions[:, 1]
    assert np.min(ypos) >= 0


def test_multiple_probes_enabled():
    # multiple probes, all enabled:

    probe = read_openephys(
        data_path / "OE_6.7_enabled_disabled_Neuropix-PXI" / "settings_enabled-enabled.xml", probe_name="ProbeA"
    )
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)

    assert probe.get_shank_count() == 1

    probe = read_openephys(
        data_path / "OE_6.7_enabled_disabled_Neuropix-PXI" / "settings_enabled-enabled.xml", probe_name="ProbeB"
    )
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 4


def test_multiple_probes_disabled():
    # multiple probes, some disabled
    probe = read_openephys(
        data_path / "OE_6.7_enabled_disabled_Neuropix-PXI" / "settings_enabled-disabled.xml", probe_name="ProbeA"
    )
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 1

    # Fail as this is disabled:
    with pytest.raises(Exception) as e:
        probe = read_openephys(
            data_path / "OE_6.7_enabled_disabled_Neuropix-PXI" / "settings_enabled-disabled.xml", probe_name="ProbeB"
        )

    assert "Inconsistency between provided probe name ProbeB and available probe ProbeA" in str(e.value)


def test_np_opto_with_sync():
    probe = read_openephys(data_path / "OE_Neuropix-PXI-opto-with-sync" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384


def test_older_than_06_format():
    ## Test with the open ephys < 0.6 format

    probe = read_openephys(data_path / "OE_5_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="100.0")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 4
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) >= 0


def test_multiple_signal_chains():
    # tests that the probe information can be loaded even if the Neuropix-PXI plugin
    # is not in the first signalchain
    probe = read_openephys(data_path / "OE_Neuropix-PXI-multiple-signalchains" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)


def test_quadbase():
    # This dataset has a Neuropixels Quad Base (4 NP2 probes on different shanks)
    for i in range(4):
        probe = read_openephys(data_path / "OE_Neuropix-PXI-QuadBase" / "settings.xml", probe_name=f"ProbeC-{i+1}")
        probe_dict = probe.to_dict(array_as_list=True)
        validate_probe_dict(probe_dict)
        assert probe.get_shank_count() == 1
        assert probe.get_contact_count() == 384
        assert set(probe.shank_ids) == set([str(i)])


def test_quadbase_custom_names():
    # This dataset has a Neuropixels Quad Base (4 NP2 probes on different shanks)
    sn = "23207205101"
    for i in range(4):
        probe = read_openephys(
            data_path / "OE_Neuropix-PXI-QuadBase" / "settings_custom_names.xml", probe_name=f"{sn}-{i+1}"
        )
        probe_dict = probe.to_dict(array_as_list=True)
        validate_probe_dict(probe_dict)
        assert probe.get_shank_count() == 1
        assert probe.get_contact_count() == 384
        assert set(probe.shank_ids) == set([str(i)])
        assert probe.name == f"{sn}-{i+1}"


def test_onebox():
    # This dataset has a Neuropixels Ultra probe with a onebox
    probe = read_openephys(data_path / "OE_OneBox-NP-Ultra" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384


def test_onix_np1():
    # This dataset has a multiple settings with different banks and configs
    probe_bankA = read_openephys(data_path / "OE_ONIX-NP" / "settings_bankA.xml")
    probe_dict = probe_bankA.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe_bankA.get_shank_count() == 1
    assert probe_bankA.get_contact_count() == 384
    assert probe_bankA.name == "PortB-Neuropixels1.0eHeadstage-Probe"
    # bank A starts at y=0 and ends at y=3820
    assert np.min(probe_bankA.contact_positions[:, 1]) == 0
    assert np.max(probe_bankA.contact_positions[:, 1]) == 3820

    probe_bankB = read_openephys(data_path / "OE_ONIX-NP" / "settings_bankB.xml")
    probe_dict = probe_bankB.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe_bankB.get_shank_count() == 1
    assert probe_bankB.get_contact_count() == 384
    assert probe_bankB.name == "PortB-Neuropixels1.0eHeadstage-Probe"
    # bank B starts at y=3840 and ends at y=7660
    assert np.min(probe_bankB.contact_positions[:, 1]) == 3840
    assert np.max(probe_bankB.contact_positions[:, 1]) == 7660

    probe_bankC = read_openephys(data_path / "OE_ONIX-NP" / "settings_bankC.xml")
    probe_dict = probe_bankC.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe_bankC.get_shank_count() == 1
    assert probe_bankC.get_contact_count() == 384
    assert probe_bankC.name == "PortB-Neuropixels1.0eHeadstage-Probe"
    # bank C starts at y=7680 and ends at y=11520
    assert np.min(probe_bankC.contact_positions[:, 1]) == 5760
    assert np.max(probe_bankC.contact_positions[:, 1]) == 9580

    # for the tetrode configuration, we expect to have 96 tetrodes
    probe_tetrodes = read_openephys(data_path / "OE_ONIX-NP" / "settings_tetrodes.xml")
    probe_dict = probe_tetrodes.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    contact_positions = probe_tetrodes.contact_positions
    distances = np.sqrt(np.sum((contact_positions[:, np.newaxis] - contact_positions[np.newaxis, :]) ** 2, axis=2))

    # For each contact, find the 3 closest neighbors (excluding itself)
    np.fill_diagonal(distances, np.inf)
    closest_indices = np.argsort(distances, axis=1)[:, :3]
    # Create tetrode groups by combining each contact with its 3 closest neighbors
    tetrode_groups = np.column_stack([np.arange(len(contact_positions)), closest_indices])
    tetrode_groups = np.sort(tetrode_groups, axis=1)
    # Find unique tetrode groups
    unique_groups = np.unique(tetrode_groups, axis=0)
    # Check that we have the expected number of tetrodes
    expected_tetrodes = len(contact_positions) // 4
    assert len(unique_groups) == expected_tetrodes, f"Expected {expected_tetrodes} tetrodes, found {len(unique_groups)}"
    # Verify each group has exactly 4 contacts
    assert unique_groups.shape[1] == 4, f"Tetrode groups should have 4 contacts, found {unique_groups.shape[1]}"


def test_onix_np2():
    # NP2.0
    probe_np2_probe0 = read_openephys(
        data_path / "OE_ONIX-NP" / "settings_NP2.xml", probe_name="PortA-Neuropixels2.0eHeadstage-Neuropixels2.0-Probe0"
    )
    probe_dict = probe_np2_probe0.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe_np2_probe0.get_contact_count() == 384
    assert probe_np2_probe0.get_shank_count() == 1

    probe_np2_probe1 = read_openephys(
        data_path / "OE_ONIX-NP" / "settings_NP2.xml", probe_name="PortA-Neuropixels2.0eHeadstage-Neuropixels2.0-Probe1"
    )
    probe_dict = probe_np2_probe1.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe_np2_probe1.get_contact_count() == 384
    assert probe_np2_probe1.get_shank_count() == 4

    for i in range(4):
        probe_0 = read_openephys(
            data_path / "OE_ONIX-NP" / f"settings_NP2_{i+1}.xml",
            probe_name=f"PortA-Neuropixels2.0eHeadstage-Neuropixels2.0-Probe0",
        )
        probe_dict = probe_0.to_dict(array_as_list=True)
        validate_probe_dict(probe_dict)
        probe_1 = read_openephys(
            data_path / "OE_ONIX-NP" / f"settings_NP2_{i+1}.xml",
            probe_name=f"PortA-Neuropixels2.0eHeadstage-Neuropixels2.0-Probe1",
        )
        probe_dict = probe_1.to_dict(array_as_list=True)
        validate_probe_dict(probe_dict)

        # all should have 384 contacts and one shank, except for i == 3, where electrodes are
        # selected on all 4 shanks
        assert probe_0.get_contact_count() == 384
        assert probe_0.get_shank_count() == 1
        assert probe_1.get_contact_count() == 384
        if i < 3:
            assert probe_1.get_shank_count() == 1
        else:
            assert probe_1.get_shank_count() == 4


def _build_probe_from_settings(settings_file, **kwargs):
    """Helper: parse settings, select probe, build from catalogue, slice."""
    probes_info = _parse_openephys_settings(settings_file)
    info = _select_openephys_probe_info(probes_info, **kwargs)
    full_probe = build_neuropixels_probe(info["probe_part_number"])
    return _slice_catalogue_probe(full_probe, info)


def test_build_openephys_probe_no_wiring():
    # Path A (SELECTED_ELECTRODES): ONIX dataset
    probe_a = _build_probe_from_settings(data_path / "OE_ONIX-NP" / "settings_bankA.xml")
    assert probe_a is not None
    assert probe_a.device_channel_indices is None

    # Path B (CHANNELS): Neuropix-PXI dataset
    probe_b = _build_probe_from_settings(data_path / "OE_Neuropix-PXI" / "settings.xml")
    assert probe_b is not None
    assert probe_b.device_channel_indices is None


def _assert_contact_ids_match_canonical_pattern(probe, label=""):
    """Assert that a probe's contact_ids are a subset of the canonical IDs from build_neuropixels_probe."""
    part_number = probe.annotations["part_number"]
    catalogue = build_neuropixels_probe(part_number)
    catalogue_ids = set(catalogue.contact_ids)
    probe_ids = set(probe.contact_ids)
    assert probe_ids.issubset(
        catalogue_ids
    ), f"{label} ({part_number}): contact_ids not in canonical pattern: {probe_ids - catalogue_ids}"


def test_read_openephys_contact_ids_match_canonical_pattern():
    """Verify that read_openephys contact_ids are consistent with SpikeGLX (issue #394).

    For each dataset, the contact_ids produced by read_openephys must be a subset of
    the contact_ids from build_neuropixels_probe(). This ensures that the same physical
    electrode gets the same contact_id regardless of acquisition system (OpenEphys vs SpikeGLX).

    The datasets from OE_Neuropix-PXI-NP-Ultra, OE_6.7_enabled_disabled_Neuropix-PXI, and
    OE_Neuropix-PXI-QuadBase were identified as inconsistent cases in PR #383
    (see https://github.com/SpikeInterface/probeinterface/pull/383#discussion_r2650588006).
    """
    # Path A (SELECTED_ELECTRODES): OE 1.0 dataset
    probe = read_openephys(data_path / "OE_1.0_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeA")
    _assert_contact_ids_match_canonical_pattern(probe, "OE_1.0 ProbeA")

    # Path B (CHANNELS): NP2 dataset (single shank)
    probe = read_openephys(data_path / "OE_Neuropix-PXI" / "settings.xml")
    _assert_contact_ids_match_canonical_pattern(probe, "NP2")

    # Path B (CHANNELS): NP2 4-shank dataset (multi-shank)
    probe = read_openephys(data_path / "OE_Neuropix-PXI-NP2-4shank" / "settings.xml")
    _assert_contact_ids_match_canonical_pattern(probe, "NP2 4-shank")

    # Path B (CHANNELS): NP-Opto dataset
    probe = read_openephys(data_path / "OE_Neuropix-PXI-opto-with-sync" / "settings.xml")
    _assert_contact_ids_match_canonical_pattern(probe, "NP-Opto")

    # Path B (CHANNELS): OneBox NP-Ultra (NP1110) dataset
    probe = read_openephys(data_path / "OE_OneBox-NP-Ultra" / "settings.xml")
    _assert_contact_ids_match_canonical_pattern(probe, "OneBox NP1110")

    # Datasets identified as inconsistent in PR #383 discussion:

    # NP-Ultra: NP1100 probes error due to catalogue mismatch (see issue #407), NP1121 should match
    probe = read_openephys(data_path / "OE_Neuropix-PXI-NP-Ultra" / "settings.xml", probe_name="ProbeD")
    _assert_contact_ids_match_canonical_pattern(probe, "NP-Ultra ProbeD")

    # enabled/disabled: NP1 and NP2014
    probe = read_openephys(
        data_path / "OE_6.7_enabled_disabled_Neuropix-PXI" / "settings_enabled-enabled.xml",
        probe_name="ProbeA",
    )
    _assert_contact_ids_match_canonical_pattern(probe, "enabled-enabled ProbeA")

    probe = read_openephys(
        data_path / "OE_6.7_enabled_disabled_Neuropix-PXI" / "settings_enabled-enabled.xml",
        probe_name="ProbeB",
    )
    _assert_contact_ids_match_canonical_pattern(probe, "enabled-enabled ProbeB")

    # QuadBase: NP2020 (4 probes)
    for i in range(4):
        probe = read_openephys(data_path / "OE_Neuropix-PXI-QuadBase" / "settings.xml", probe_name=f"ProbeC-{i+1}")
        _assert_contact_ids_match_canonical_pattern(probe, f"QuadBase ProbeC-{i+1}")


def test_read_openephys_backward_compatible():
    # Verify read_openephys still produces valid probes with device_channel_indices set
    # Path B dataset
    probe = read_openephys(data_path / "OE_Neuropix-PXI" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.device_channel_indices is not None
    assert len(probe.device_channel_indices) == probe.get_contact_count()
    assert np.array_equal(probe.device_channel_indices, np.arange(probe.get_contact_count()))

    # Path A dataset
    probe_a = read_openephys(data_path / "OE_1.0_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeA")
    probe_dict = probe_a.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe_a.device_channel_indices is not None
    assert np.array_equal(probe_a.device_channel_indices, np.arange(probe_a.get_contact_count()))


def _read_oebin_electrode_indices(oebin_file, stream_name):
    """Read electrode_index metadata from an oebin file for a given stream."""
    with open(oebin_file) as f:
        oebin = json.load(f)
    for cs in oebin.get("continuous", []):
        folder_name = cs.get("folder_name", "")
        if stream_name in folder_name or folder_name in stream_name:
            indices = []
            for ch in cs.get("channels", []):
                for m in ch.get("channel_metadata", []):
                    if m.get("name") == "electrode_index":
                        indices.append(m["value"][0])
            return indices
    return []


def test_read_openephys_with_oebin_wiring():
    """Verify wiring invariant: for each contact, the oebin's electrode_index at the
    assigned binary column matches the contact's electrode index."""
    settings = data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "settings.xml"
    oebin = (
        data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "experiment1" / "recording1" / "structure.oebin"
    )
    stream_name = "Neuropix-PXI-100.ProbeA"

    probe = read_openephys(settings, stream_name=stream_name, oebin_file=oebin)

    assert probe.get_contact_count() == 384
    assert probe.device_channel_indices is not None

    # Wiring invariant
    oebin_electrode_indices = _read_oebin_electrode_indices(oebin, stream_name)
    for i, contact_id in enumerate(probe.contact_ids):
        electrode_index = int(contact_id.split("e")[-1])
        column = probe.device_channel_indices[i]
        assert oebin_electrode_indices[column] == electrode_index, (
            f"Contact {i} ({contact_id}): expected electrode_index {electrode_index} "
            f"at column {column}, got {oebin_electrode_indices[column]}"
        )


def test_read_openephys_with_oebin_contact_ids_match_canonical_pattern():
    """Verify that contact_ids with oebin are consistent with SpikeGLX (issue #394)."""
    # NP2014 single-shank
    probe = read_openephys(
        data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "settings.xml",
        stream_name="Neuropix-PXI-100.ProbeA",
        oebin_file=data_path
        / "OE_Neuropix-PXI-NP1-binary"
        / "Record_Node_101"
        / "experiment1"
        / "recording1"
        / "structure.oebin",
    )
    _assert_contact_ids_match_canonical_pattern(probe, "NP2014 binary")

    # NP1032 4-shank
    probe = read_openephys(
        data_path / "OE_Neuropix-PXI-NP2-4shank-binary" / "Record_Node_101" / "settings.xml",
        stream_name="Neuropix-PXI-100.ProbeA-AP",
        oebin_file=data_path
        / "OE_Neuropix-PXI-NP2-4shank-binary"
        / "Record_Node_101"
        / "experiment4"
        / "recording2"
        / "structure.oebin",
    )
    _assert_contact_ids_match_canonical_pattern(probe, "NP1032 binary")


def test_read_openephys_with_oebin_sync_channel_filtered():
    """Verify that the oebin sync channel (385 channels) is filtered, producing 384 contacts."""
    settings = data_path / "OE_Neuropix-PXI-NP2-4shank-binary" / "Record_Node_101" / "settings.xml"
    oebin = (
        data_path
        / "OE_Neuropix-PXI-NP2-4shank-binary"
        / "Record_Node_101"
        / "experiment4"
        / "recording2"
        / "structure.oebin"
    )

    probe = read_openephys(settings, stream_name="Neuropix-PXI-100.ProbeA-AP", oebin_file=oebin)
    assert probe.get_contact_count() == 384


def test_read_openephys_with_oebin_plugin_channel_key():
    """Verify that plugin_channel_key annotation is set when using oebin_file."""
    settings = data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "settings.xml"
    oebin = (
        data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "experiment1" / "recording1" / "structure.oebin"
    )
    stream_name = "Neuropix-PXI-100.ProbeA"

    probe = read_openephys(settings, stream_name=stream_name, oebin_file=oebin)
    keys = probe.contact_annotations.get("plugin_channel_key", None)
    assert keys is not None, "plugin_channel_key annotation not set"
    assert len(keys) == probe.get_contact_count()
    assert all(k.startswith("CH") for k in keys)


def test_read_openephys_with_oebin_no_matching_stream():
    """Verify error when stream_name doesn't match any probe in settings."""
    settings = data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "settings.xml"
    oebin = (
        data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "experiment1" / "recording1" / "structure.oebin"
    )

    with pytest.raises(Exception, match="Inconsistency between provided stream"):
        read_openephys(settings, stream_name="NonExistentStream", oebin_file=oebin)


def test_read_openephys_oebin_file_requires_stream_name():
    """Verify ValueError when oebin_file is provided without stream_name."""
    settings = data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "settings.xml"
    oebin = (
        data_path / "OE_Neuropix-PXI-NP1-binary" / "Record_Node_101" / "experiment1" / "recording1" / "structure.oebin"
    )
    with pytest.raises(ValueError, match="stream_name is required"):
        read_openephys(settings, oebin_file=oebin)


def test_read_openephys_multishank_wiring():
    """Verify that multi-shank wiring correctly uses global electrode indices.

    This test uses an NP2013 (4-shank) dataset where electrode_index values in
    the oebin are global (0-5119). The old code extracted shank-local IDs from
    contact_ids like 's3e0' -> 0, which was wrong. The fix computes the global
    index as shank_id * electrodes_per_shank + local_id (e.g. 3 * 1280 + 0 = 3840).
    """
    settings = data_path / "OE_Neuropix-PXI-NP2-multishank-binary" / "Record_Node_109" / "settings.xml"
    oebin = (
        data_path
        / "OE_Neuropix-PXI-NP2-multishank-binary"
        / "Record_Node_109"
        / "experiment1"
        / "recording1"
        / "structure.oebin"
    )
    stream_name = "Neuropix-PXI-103.ProbeA"

    probe = read_openephys(settings, stream_name=stream_name, oebin_file=oebin)

    assert probe.get_contact_count() == 384
    assert probe.device_channel_indices is not None

    # Wiring invariant: for each contact, the oebin's electrode_index at the
    # assigned binary column must match the contact's global electrode index.
    oebin_electrode_indices = _read_oebin_electrode_indices(oebin, stream_name)

    from probeinterface.neuropixels_tools import _contact_id_to_global_electrode_index

    # NP2013: 2 cols * 640 rows = 1280 electrodes per shank
    electrodes_per_shank = 1280

    for i, contact_id in enumerate(probe.contact_ids):
        global_electrode_index = _contact_id_to_global_electrode_index(contact_id, electrodes_per_shank)
        column = probe.device_channel_indices[i]
        assert oebin_electrode_indices[column] == global_electrode_index, (
            f"Contact {i} ({contact_id}): expected global electrode_index {global_electrode_index} "
            f"at column {column}, got {oebin_electrode_indices[column]}"
        )


if __name__ == "__main__":
    # test_multiple_probes()
    # test_NP_Ultra()
    test_onix_np1()
    test_onix_np2()
