from pathlib import Path
import numpy as np

import pytest

from probeinterface import (
    read_spikeglx,
    parse_spikeglx_meta,
    get_saved_channel_indices_from_spikeglx_meta,
)

data_path = Path(__file__).absolute().parent.parent / "data" / "spikeglx"


def test_NP1():
    probe = read_spikeglx(data_path / "Noise_g0_t0.imec0.ap.meta")
    assert "1.0" in probe.annotations["name"]


def test_NP2_4_shanks():
    probe = read_spikeglx(data_path / "TEST_20210920_0_g0_t0.imec0.ap.meta")
    assert probe.get_shank_count() == 4
    assert "2.0" in probe.annotations["name"]


def test_NP2_1_shanks():
    probe = read_spikeglx(data_path / "p2_g0_t0.imec0.ap.meta")
    assert "2.0" in probe.annotations["name"]
    assert probe.get_shank_count() == 1


def test_NP_phase3A():
    # Data provided by rtraghavan
    probe = read_spikeglx(data_path / "NeuropixelPhase3A_file_g0_t0.imec.ap.meta")
    assert "Phase3a" in probe.annotations["name"]
    assert probe.get_shank_count() == 1


def test_NP2_4_shanks():
    # Data provided by Jennifer Colonell
    probe = read_spikeglx(data_path / "NP24_g0_t0.imec0.ap.meta")
    assert "2.0" in probe.annotations["name"]
    assert probe.get_shank_count() == 4


def test_NP1_large_depth_sapn():
    # Data provided by Tom Bugnon NP1 with large Depth span
    probe = read_spikeglx(data_path / "allan-longcol_g0_t0.imec0.ap.meta")
    assert "1.0" in probe.annotations["name"]
    assert probe.get_shank_count() == 1
    ypos = probe.contact_positions[:, 1]
    assert (np.max(ypos) - np.min(ypos)) > 7600


def test_NP1_other_example():
    # Data provided by Tom Bugnon NP1
    probe = read_spikeglx(data_path / "doppio-checkerboard_t0.imec0.ap.meta")
    print(probe)
    assert "1.0" in probe.annotations["name"]
    assert probe.get_shank_count() == 1
    ypos = probe.contact_positions[:, 1]
    assert (np.max(ypos) - np.min(ypos)) > 7600


def tes_NP1_384_channels():
    # example by Pierre Yger NP1.0 with 384 but only 151 channels are saved
    probe = read_spikeglx(data_path / "Day_3_g0_t0.imec1.ap.meta")
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 151
    assert 152 not in probe.contact_annotations["channel_ids"]


def test_parse_meta():
    for meta_file in [
        "doppio-checkerboard_t0.imec0.ap.meta",
        "Day_3_g0_t0.imec1.ap.meta",
        "allan-longcol_g0_t0.imec0.ap.meta",
    ]:
        meta = parse_spikeglx_meta(data_path / meta_file)


def test_get_saved_channel_indices_from_spikeglx_meta():
    # all channel saved + 1 synchro
    chan_inds = get_saved_channel_indices_from_spikeglx_meta(
        data_path / "Noise_g0_t0.imec0.ap.meta"
    )
    assert chan_inds.size == 385

    # example by Pierre Yger NP1.0 with 384 but only 151 channels are saved + 1 synchro
    chan_inds = get_saved_channel_indices_from_spikeglx_meta(
        data_path / "Day_3_g0_t0.imec1.ap.meta"
    )
    assert chan_inds.size == 152
