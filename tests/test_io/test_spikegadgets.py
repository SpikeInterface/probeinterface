from pathlib import Path
from xml.etree import ElementTree

from probeinterface import read_spikegadgets
from probeinterface.io import parse_spikegadgets_header
from probeinterface.testing import validate_probe_dict

data_path = Path(__file__).absolute().parent.parent / "data" / "spikegadgets"
test_file = "SpikeGadgets_test_data_2xNpix1.0_20240318_173658_header_only.rec"
test_file_np2_4shank = "SpikeGadgets_test_data_NP2_4shank_20260122_header_only.rec"


def test_parse_meta():
    header_txt = parse_spikegadgets_header(data_path / test_file)
    root = ElementTree.fromstring(header_txt)
    assert root.find("GlobalConfiguration") is not None
    assert root.find("HardwareConfiguration") is not None
    assert root.find("SpikeConfiguration") is not None


def test_neuropixels_1_reader():
    probe_group = read_spikegadgets(data_path / test_file, raise_error=False)
    assert len(probe_group.probes) == 2
    for probe in probe_group.probes:
        probe_dict = probe.to_dict(array_as_list=True)
        validate_probe_dict(probe_dict)
        assert probe.model_name == ""
        assert probe.get_shank_count() == 1
        assert probe.get_contact_count() == 384
    assert probe_group.get_contact_count() == 768


def test_neuropixels_2_4shank_reader():
    probe_group = read_spikegadgets(data_path / test_file_np2_4shank, raise_error=False)
    assert len(probe_group.probes) == 1
    probe = probe_group.probes[0]
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.model_name == ""
    assert probe.get_contact_count() == 384
    assert probe.device_channel_indices.shape == (384,)
    # This specific recording only activates electrodes on a single shank, so we
    # assert the sliced probe has at least one shank from the 4-shank catalogue
    # and that all active contact ids follow the s<shank>e<electrode> scheme.
    assert 1 <= probe.get_shank_count() <= 4
    assert all(cid.startswith("s") and "e" in cid for cid in probe.contact_ids)


if __name__ == "__main__":
    test_parse_meta()
    test_neuropixels_1_reader()
    test_neuropixels_2_4shank_reader()
