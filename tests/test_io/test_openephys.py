from pathlib import Path

import numpy as np

import pytest

from probeinterface import read_openephys

data_path = Path(__file__).absolute().parent.parent / "data" / "openephys"


def test_NP2():
    # NP2
    probe = read_openephys(data_path / "OE_Neuropix-PXI" / "settings.xml")
    assert probe.get_shank_count() == 1
    assert "2.0 - Single Shank" in probe.model_name


def test_NP1_subset():
    # NP1 - 200 channels selected by recording_state in Record Node
    probe_ap = read_openephys(
        data_path / "OE_Neuropix-PXI-subset" / "settings.xml", stream_name="ProbeA-AP"
    )

    assert probe_ap.get_shank_count() == 1
    assert "1.0" in probe_ap.model_name
    assert probe_ap.get_contact_count() == 200

    probe_lf = read_openephys(
        data_path / "OE_Neuropix-PXI-subset" / "settings.xml", stream_name="ProbeA-LFP"
    )

    assert probe_lf.get_shank_count() == 1
    assert "1.0" in probe_lf.model_name
    assert len(probe_lf.contact_positions) == 200

    # Not specifying the stream_name should raise an Exception, because both the ProbeA-AP and
    # ProbeA-LFP have custome channel selections
    with pytest.raises(AssertionError):
        probe = read_openephys(data_path / "OE_Neuropix-PXI-subset" / "settings.xml")


def test_multiple_probes():
    # multiple probes
    probeA = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeA"
    )

    assert probeA.get_shank_count() == 1
    assert "1.0" in probeA.model_name

    probeB = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
        stream_name="RecordNode#ProbeB",
    )

    assert probeB.get_shank_count() == 1

    probeC = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
        serial_number="20403311714",
    )

    assert probeC.get_shank_count() == 1

    probeD = read_openephys(
        data_path / "OE_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeD"
    )

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
    assert "2.0 - Multishank" in probeB2.model_name

    ypos = probeB2.contact_positions[:, 1]
    assert np.min(ypos) >= 0


def test_np_otpo_with_sync():
    probe = read_openephys(data_path / "OE_Neuropix-PXI-with-sync" / "settings.xml")
    assert probe.model_name == "Neuropixels Opto"
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384



def test_older_than_06_format():
    ## Test with the open ephys < 0.6 format

    probe = read_openephys(
        data_path / "OE_5_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="100.0"
    )

    assert probe.get_shank_count() == 4
    assert "2.0 - Multishank" in probe.model_name
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) >= 0


if __name__ == "__main__":
    test_multiple_probes()
    test_older_than_06_format()
