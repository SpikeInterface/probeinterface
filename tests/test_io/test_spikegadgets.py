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
test_file_np2_4shank = "SpikeGadgets_test_data_NP2_4shank_20260122_header_only.rec"


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


def test_neuropixels_2_4shank_reader():
    # This NP2.0 4-shank fixture activates 48 rows of electrodes across all 4
    # shanks (probeColumns 0-7), so it exercises the row-major-to-shank-major
    # chind remapping defined in `_spikegadgets_chind_np2_4shank`. The recovered
    # ml coordinates should match the SpikeChannel coord_ml values up to a
    # single stereotactic offset (the workspace-baked probe origin).
    import numpy as np
    from xml.etree import ElementTree as ET

    probe_group = read_spikegadgets_neuropixels(data_path / test_file_np2_4shank, raise_error=False)
    assert len(probe_group.probes) == 1
    probe = probe_group.probes[0]
    probe_dict = probe.to_dict(array_as_list=True)
    validate_probe_dict(probe_dict)
    assert probe.model_name == ""
    assert probe.get_contact_count() == 384
    assert probe.device_channel_indices.shape == (384,)
    assert probe.get_shank_count() == 4
    # Each shank should contribute 96 contacts (48 rows × 2 cols per shank).
    shank_ids = np.array(probe.shank_ids)
    for shank in ("0", "1", "2", "3"):
        assert (shank_ids == shank).sum() == 96, f"shank {shank} contact count"
    assert all(cid.startswith("s") and "e" in cid for cid in probe.contact_ids)

    # Verify catalogue positions are consistent with .rec coord_ml/coord_dv up
    # to a single stereotactic offset shared across all electrodes.
    header_txt = parse_spikegadgets_header(data_path / test_file_np2_4shank)
    root = ET.fromstring(header_txt)
    sconf = root.find("SpikeConfiguration")
    rec_positions = {}
    for ntrode in sconf:
        chind = int(ntrode.attrib["id"][1:]) - 1
        ch = ntrode.find("SpikeChannel")
        rec_positions[chind] = (float(ch.attrib["coord_ml"]), float(ch.attrib["coord_dv"]))
    # Sample 1: chind 1671 should land on s0e416 (shank 0, ml=0, dv=3120 in catalogue).
    sample_chind = 1671
    ml_rec, dv_rec = rec_positions[sample_chind]
    sample_idx_in_probe = list(probe.contact_ids).index("s0e416")
    ml_cat, dv_cat = probe.contact_positions[sample_idx_in_probe]
    offset_ml = ml_rec - ml_cat
    offset_dv = dv_rec - dv_cat
    # Sample 2: chind 1664 should land on s3e417.
    ml_rec_2, dv_rec_2 = rec_positions[1664]
    sample_idx_2 = list(probe.contact_ids).index("s3e417")
    ml_cat_2, dv_cat_2 = probe.contact_positions[sample_idx_2]
    assert abs((ml_rec_2 - ml_cat_2) - offset_ml) < 1e-6, "ml offset must be constant across shanks"
    assert abs((dv_rec_2 - dv_cat_2) - offset_dv) < 1e-6, "dv offset must be constant across rows"


def test_stereotactic_annotations_np1():
    # SpikeChannel coord_ml/dv/ap from the .rec are stored as per-contact
    # annotations on the output probe. Sentinel: chind 383 (id "1384" on probe
    # 1) maps to catalogue idx 383 (e383) under identity remap; the matching
    # SpikeChannel has coord_ml="-8" coord_dv="3920" coord_ap="0".
    probe_group = read_spikegadgets_neuropixels(data_path / test_file)
    probe = probe_group.probes[0]
    n_contacts = probe.get_contact_count()
    for key in ("stereotactic_ml", "stereotactic_dv", "stereotactic_ap"):
        assert key in probe.contact_annotations
        assert probe.contact_annotations[key].shape == (n_contacts,)
    i = list(probe.contact_ids).index("e383")
    assert probe.contact_annotations["stereotactic_ml"][i] == -8.0
    assert probe.contact_annotations["stereotactic_dv"][i] == 3920.0
    assert probe.contact_annotations["stereotactic_ap"][i] == 0.0


def test_stereotactic_annotations_np2_4shank():
    # Same check for NP2.0 4-shank: chind 1671 maps to catalogue idx 416
    # (s0e416) via the row-major-to-shank-major remap; the matching SpikeChannel
    # has coord_ml="-383" coord_dv="3295" coord_ap="0".
    probe_group = read_spikegadgets_neuropixels(data_path / test_file_np2_4shank)
    probe = probe_group.probes[0]
    n_contacts = probe.get_contact_count()
    for key in ("stereotactic_ml", "stereotactic_dv", "stereotactic_ap"):
        assert key in probe.contact_annotations
        assert probe.contact_annotations[key].shape == (n_contacts,)
    i = list(probe.contact_ids).index("s0e416")
    assert probe.contact_annotations["stereotactic_ml"][i] == -383.0
    assert probe.contact_annotations["stereotactic_dv"][i] == 3295.0
    assert probe.contact_annotations["stereotactic_ap"][i] == 0.0


def test_has_spikegadgets_neuropixels_probes_np2():
    # NP2.0 4-shank .rec should also report True.
    assert has_spikegadgets_neuropixels_probes(data_path / test_file_np2_4shank) is True


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
    test_neuropixels_2_4shank_reader()
    test_stereotactic_annotations_np1()
    test_stereotactic_annotations_np2_4shank()
    test_has_spikegadgets_neuropixels_probes_np2()
    test_read_spikegadgets_deprecation_warning()
    test_has_spikegadgets_neuropixels_probes_positive()
    test_has_spikegadgets_neuropixels_probes_missing_file()
