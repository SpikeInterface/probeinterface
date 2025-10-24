import glob
from pathlib import Path
import numpy as np

import pytest

from probeinterface import (
    read_spikeglx,
    parse_spikeglx_meta,
    get_saved_channel_indices_from_spikeglx_meta,
    parse_spikeglx_snsGeomMap,
)
from probeinterface.testing import validate_probe_dict

data_path = Path(__file__).absolute().parent.parent / "data" / "spikeglx"
meta_files = glob.glob(str(data_path / "*.meta"))


@pytest.mark.parametrize("meta_file", meta_files)
def test_valid_probe_dict(meta_file: str):
    probe = read_spikeglx(data_path / meta_file)
    validate_probe_dict(probe.to_dict(array_as_list=True))


def test_parse_meta():
    for meta_file in [
        "doppio-checkerboard_t0.imec0.ap.meta",
        "NP1_saved_only_subset_of_channels.meta",
        "allan-longcol_g0_t0.imec0.ap.meta",
    ]:
        meta = parse_spikeglx_meta(data_path / meta_file)


def test_get_saved_channel_indices_from_spikeglx_meta():
    # all channel saved + 1 synchro
    chan_inds = get_saved_channel_indices_from_spikeglx_meta(data_path / "Noise_g0_t0.imec0.ap.meta")
    assert chan_inds.size == 385

    # example by Pierre Yger NP1.0 with 384 but only 151 channels are saved + 1 synchro
    chan_inds = get_saved_channel_indices_from_spikeglx_meta(data_path / "NP1_saved_only_subset_of_channels.meta")
    assert chan_inds.size == 152


def test_NP1():
    probe = read_spikeglx(data_path / "Noise_g0_t0.imec0.ap.meta")
    assert "1.0" in probe.description


def test_NP_phase3A():
    # Data provided by rtraghavan
    probe = read_spikeglx(data_path / "phase3a.imec.ap.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384

    # Test contact geometry
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)


def test_NP2_1_shanks():
    probe = read_spikeglx(data_path / "p2_g0_t0.imec0.ap.meta")
    assert "2.0" in probe.description
    assert probe.get_shank_count() == 1


def test_NP2_4_shanks():
    probe = read_spikeglx(data_path / "NP2_4_shanks.imec0.ap.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    assert probe.get_shank_count() == 4
    assert probe.get_contact_count() == 384

    # Test contact geometry
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    # This file does not save the channels from 0 as the one above (NP2_4_shanks_g0_t0.imec0.ap.meta)
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) == pytest.approx(0)


def test_NP2_2013_all():
    # Data provided by Jennifer Colonell
    probe = read_spikeglx(data_path / "NP2_2013_all_channels.imec0.ap.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    # all channels are from the first shank
    assert probe.get_shank_count() == 4
    assert probe.get_contact_count() == 384

    # Test contact geometry
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    # This file does not save the channels from 0 as the one above (NP2_4_shanks_g0_t0.imec0.ap.meta)
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) == pytest.approx(0)


def test_NP2_2013_subset():
    # Data provided by Jennifer Colonell
    probe = read_spikeglx(data_path / "NP2_2013_subset_channels.imec0.ap.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    # all channels are from the first shank
    assert probe.get_shank_count() == 4
    assert probe.get_contact_count() == 120

    # Test contact geometry
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    # This file does not save the channels from 0 as the one above (NP2_4_shanks_g0_t0.imec0.ap.meta)
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) == pytest.approx(0)


def test_NP2_4_shanks_with_different_electrodes_saved():
    # Data provided by Jennifer Colonell
    probe = read_spikeglx(data_path / "NP2_4_shanks_save_different_electrodes.imec0.ap.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    assert probe.get_shank_count() == 4
    assert probe.get_contact_count() == 384

    # Test contact geometry
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    # This file does not save the channels from 0 as the one above (NP2_4_shanks_g0_t0.imec0.ap.meta)
    ypos = probe.contact_positions[:, 1]
    assert np.min(ypos) == pytest.approx(4080.0)
    assert np.max(ypos) == pytest.approx(4785.0)


def test_NP1_large_depth_span():
    # Data provided by Tom Bugnon NP1 with large Depth span
    probe = read_spikeglx(data_path / "allan-longcol_g0_t0.imec0.ap.meta")
    assert "1.0" in probe.description
    assert probe.get_shank_count() == 1
    ypos = probe.contact_positions[:, 1]
    assert (np.max(ypos) - np.min(ypos)) > 7600


def test_NP1_other_example():
    # Data provided by Tom Bugnon NP1
    probe = read_spikeglx(data_path / "doppio-checkerboard_t0.imec0.ap.meta")
    print(probe)
    assert "1.0" in probe.description
    assert probe.get_shank_count() == 1
    ypos = probe.contact_positions[:, 1]
    assert (np.max(ypos) - np.min(ypos)) > 7600


def tes_NP1_384_channels():
    # example by Pierre Yger NP1.0 with 384 but only 151 channels are saved
    probe = read_spikeglx(data_path / "NP1_saved_only_subset_of_channels.meta")
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 151
    assert 152 not in probe.contact_annotations["channel_ids"]


def test_NPH_long_staggered():
    # Data provided by Nate Dolensek
    probe = read_spikeglx(data_path / "non_human_primate_long_staggered.imec0.ap.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384

    # Test contact geometry
    x_pitch = 56.0
    y_pitch = 20.0
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    contact_positions = probe.contact_positions
    x = contact_positions[:, 0]
    y = contact_positions[:, 1]

    # Every second contact the x position should increase by x_pitch
    increase = np.diff(x)
    every_second_increase = increase[::2]
    x_pitch = 87
    assert np.allclose(every_second_increase, x_pitch)

    # Every second contact should be staggered by contact_width
    every_second_contact = x[::2]
    staggered_values = np.abs(np.diff(every_second_contact))
    contact_width = 16
    assert np.allclose(staggered_values, contact_width)

    # Every second contact should increase by y_pitch
    y_pitch = 20
    every_second_contact = y[::2]
    increase = np.diff(every_second_contact)
    assert np.allclose(increase, y_pitch)

    # Test vector annotaitons
    ap_gains = probe.contact_annotations["ap_gains"]
    lf_gains = probe.contact_annotations["lf_gains"]
    banks = probe.contact_annotations["banks"]
    references = probe.contact_annotations["references"]
    filters = probe.contact_annotations["ap_hp_filters"]
    assert np.allclose(ap_gains, 500)
    assert np.allclose(lf_gains, 250)
    assert np.allclose(banks, 0)
    assert np.allclose(references, 0)
    assert np.allclose(filters, 1)


def test_NPH_short_linear_probe_type_0():
    # Data provided by Jonathan A Michaels
    probe = read_spikeglx(data_path / "non_human_primate_short_linear_probe_type_0.meta")

    assert probe.manufacturer == "imec"

    assert probe.ndim == 2
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 384

    # Test contact geometry
    x_pitch = 56.0
    y_pitch = 20.0
    contact_width = 12.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    contact_positions = probe.contact_positions
    x = contact_positions[:, 0]
    y = contact_positions[:, 1]

    # Every second contact the x position should increase by x_pitch
    increase = np.diff(x)
    every_second_increase = increase[::2]
    x_pitch = 32
    assert np.allclose(every_second_increase, x_pitch)

    # Every second contact should increase by y_pitch
    y_pitch = 20.0
    every_second_contact = y[::2]
    increase = np.diff(every_second_contact)
    assert np.allclose(increase, y_pitch)

    # Test vector annotaitons
    ap_gains = probe.contact_annotations["ap_gains"]
    lf_gains = probe.contact_annotations["lf_gains"]
    banks = probe.contact_annotations["banks"]
    references = probe.contact_annotations["references"]
    filters = probe.contact_annotations["ap_hp_filters"]
    assert np.allclose(ap_gains, 500)
    assert np.allclose(lf_gains, 250)
    assert np.allclose(banks, 0)
    assert np.allclose(references, 0)
    assert np.allclose(filters, 1)


def test_ultra_probe():
    # Data provided by Alessio
    probe = read_spikeglx(data_path / "npUltra.meta")

    assert probe.manufacturer == "imec"

    # Test contact geometry
    contact_width = 5.0
    contact_shape = "square"

    assert np.all(probe.contact_shape_params == {"width": contact_width})
    assert np.all(probe.contact_shapes == contact_shape)

    contact_positions = probe.contact_positions
    x = contact_positions[:, 0]
    y = contact_positions[:, 1]

    expected_electrode_columns = 8
    unique_x_values = np.unique(x)
    assert unique_x_values.size == expected_electrode_columns

    expected_electode_rows = 48
    unique_y_values = np.unique(y)
    assert unique_y_values.size == expected_electode_rows


def test_CatGT_NP1():
    probe = read_spikeglx(data_path / "catgt.meta")
    assert "1.0" in probe.description


def test_snsGeomMap():
    # check when snsGeomMap is present if contact positions are the same from imroTbl

    for meta_file in data_path.glob("*.meta"):

        meta = parse_spikeglx_meta(meta_file)
        if "snsGeomMap" in meta:
            num_shank, shank_width, shank_pitch, shank_ids, x_pos, y_pos, activated = parse_spikeglx_snsGeomMap(meta)

            # this is read from imroTbl internally
            probe = read_spikeglx(meta_file)
            assert probe.get_shank_count() == num_shank
            assert probe.get_contact_count() == x_pos.size
            probe_x = probe.contact_positions[:, 0]
            probe_y = probe.contact_positions[:, 1]

            # for x coordinate : we have a shift between spikeglx and probeinterface (the reference is not the same)
            # for y coordinate : it is the same
            # in parse_spikeglx_snsGeomMap x_pos, y_pos are referenced per shank
            # but after the offset this should equal per shank
            for shank_id in np.unique(shank_ids):
                mask = shank_ids == shank_id
                x_offset = x_pos[mask][0] - probe_x[mask][0]
                assert np.array_equal(probe_x[mask] + x_offset, x_pos[mask])
                assert np.array_equal(probe_y[mask], y_pos[mask])


if __name__ == "__main__":
    # test_NP2_1_shanks()
    test_snsGeomMap()
