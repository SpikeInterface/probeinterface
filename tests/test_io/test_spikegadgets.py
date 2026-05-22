from pathlib import Path
from xml.etree import ElementTree

import pytest

from probeinterface import (
    read_spikegadgets,
    read_spikegadgets_neuropixels,
    has_spikegadgets_neuropixels_probes,
)
from probeinterface.io import parse_spikegadgets_header
from probeinterface.testing import validate_probe_dict

data_path = Path(__file__).absolute().parent.parent / "data" / "spikegadgets"
test_file = "SpikeGadgets_test_data_2xNpix1.0_20240318_173658_header_only.rec"


def test_parse_meta():
    header_txt = parse_spikegadgets_header(data_path / test_file)
    root = ElementTree.fromstring(header_txt)
    assert root.find("GlobalConfiguration") is not None
    assert root.find("HardwareConfiguration") is not None
    assert root.find("SpikeConfiguration") is not None


def test_neuropixels_1_reader():
    probe_group = read_spikegadgets_neuropixels(data_path / test_file, raise_error=False)
    assert len(probe_group.probes) == 2
    for probe in probe_group.probes:
        probe_dict = probe.to_dict(array_as_list=True)
        validate_probe_dict(probe_dict)
        assert probe.model_name == ""
        assert probe.get_shank_count() == 1
        assert probe.get_contact_count() == 384
    assert probe_group.get_contact_count() == 768


def test_read_spikegadgets_deprecation_warning():
    # Old read_spikegadgets name must still work but emit DeprecationWarning pointing at the new name.
    with pytest.warns(DeprecationWarning, match="read_spikegadgets_neuropixels"):
        read_spikegadgets(data_path / test_file, raise_error=False)


def test_has_spikegadgets_neuropixels_probes_positive():
    # A real Neuropixels .rec header should report True.
    assert has_spikegadgets_neuropixels_probes(data_path / test_file) is True


def test_has_spikegadgets_neuropixels_probes_missing_file():
    # Unreadable / nonexistent files return False rather than raising.
    assert has_spikegadgets_neuropixels_probes(data_path / "does_not_exist.rec") is False


if __name__ == "__main__":
    test_parse_meta()
    test_neuropixels_1_reader()
    test_read_spikegadgets_deprecation_warning()
    test_has_spikegadgets_neuropixels_probes_positive()
    test_has_spikegadgets_neuropixels_probes_missing_file()
