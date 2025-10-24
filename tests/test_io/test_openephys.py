from pathlib import Path

import numpy as np

import pytest

from probeinterface import read_openephys
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
    # This dataset has 4 NP-Ultra probes (3 type 1, 1 type 2)
    probeA = read_openephys(
        data_path / "OE_Neuropix-PXI-NP-Ultra" / "settings.xml",
        probe_name="ProbeA",
    )
    probe_dict = probeA.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probeA.get_shank_count() == 1
    assert probeA.get_contact_count() == 384

    probeB = read_openephys(
        data_path / "OE_Neuropix-PXI-NP-Ultra" / "settings.xml",
        probe_name="ProbeB",
    )
    probe_dict = probeB.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probeB.get_shank_count() == 1
    assert probeB.get_contact_count() == 384

    probeF = read_openephys(
        data_path / "OE_Neuropix-PXI-NP-Ultra" / "settings.xml",
        probe_name="ProbeF",
    )
    probe_dict = probeF.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probeF.get_shank_count() == 1
    assert probeF.get_contact_count() == 384

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


if __name__ == "__main__":
    # test_multiple_probes()
    # test_NP_Ultra()
    test_onix_np1()
    test_onix_np2()
