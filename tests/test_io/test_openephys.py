from pathlib import Path

import numpy as np

import pytest

from probeinterface import read_openephys

data_folder = Path(__file__).absolute().parent.parent


def test_NP1():
    # NP1
    probe = read_openephys(data_folder / "OE_Neuropix-PXI" / "settings.xml")
    assert probe.get_shank_count() == 1
    assert "2.0 - Single Shank" in probe.annotations["name"]

def test_multiple_probes():
    # multiple probes
    probeA = read_openephys(
        data_folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeA"
    )

    assert probeA.get_shank_count() == 1
    assert "1.0" in probeA.annotations["name"]

    probeB = read_openephys(
        data_folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
        stream_name="RecordNode#ProbeB",
    )

    assert probeB.get_shank_count() == 1

    probeC = read_openephys(
        data_folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
        serial_number="20403311714",
    )

    assert probeC.get_shank_count() == 1

    probeD = read_openephys(
        data_folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="ProbeD"
    )

    assert probeD.get_shank_count() == 1

    assert probeA.annotations["probe_serial_number"] == "17131307831"
    assert probeB.annotations["probe_serial_number"] == "20403311724"
    assert probeC.annotations["probe_serial_number"] == "20403311714"
    assert probeD.annotations["probe_serial_number"] == "21144108671"

    probeA2 = read_openephys(
        data_folder / "OE_Neuropix-PXI-multi-probe" / "settings_2.xml", probe_name="ProbeA"
    )

    assert probeA2.get_shank_count() == 1
    ypos = probeA2.contact_positions[:, 1]
    assert np.min(ypos) >= 0

    probeB2 = read_openephys(
        data_folder / "OE_Neuropix-PXI-multi-probe" / "settings_2.xml", probe_name="ProbeB"
    )

    assert probeB2.get_shank_count() == 1
    assert "2.0 - Multishank" in probeB2.annotations["name"]

    ypos = probeB2.contact_positions[:, 1]
    assert np.min(ypos) >= 0


def test_older_than_06_format():
    ## Test with the open ephys < 0.6 format

    probe = read_openephys(
        data_folder / "OE_5_Neuropix-PXI-multi-probe" / "settings.xml", probe_name="100.0"
    )

    assert probe.get_shank_count() == 4
    assert "2.0 - Multishank" in probe.annotations["name"]
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) >= 0

