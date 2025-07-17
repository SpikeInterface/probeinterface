from pathlib import Path

import numpy as np
import glob

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
    # ProbeA-LFP have custome channel selections
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


def test_onebox():
    # This dataset has a Neuropixels Ultra probe with a onebox
    probe = read_openephys(data_path / "OE_OneBox-NP-Ultra" / "settings.xml")
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384


if __name__ == "__main__":
    # test_multiple_probes()
    # test_NP_Ultra()
    test_multiple_signal_chains()
