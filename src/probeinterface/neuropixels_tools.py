"""
Here we implement probe info related to Neuropixels.
Both spikeglx (using meta file) and openephys with neuropixel (using xml file) is handled.

Note:
  * the centre of the first left columns and the first bottom row is our reference (x=0, y=0)

"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import warnings
from packaging.version import parse

import numpy as np

from .probe import Probe
from .utils import import_safely


# this dict define the contour for one shank (duplicated when several shanks so)
# note that a final "contour_shift" is applied
polygon_contour_description = {
    # NP1 and NP2 (1 and 4 shanks)
    "np70": [
        (0, 10000),
        (0, 0),
        (35, -175),
        (70, 0),
        (70, 10000),
    ],
    "nhp90": [
        (0, 10000),
        (0, 0),
        (45, -342),
        (90, 0),
        (90, 10000),
    ],
    "nhp125": [
        (0, 10000),
        (0, 0),
        (62.5, -342),
        (125, 0),
        (125, 10000),
    ],
}


# IMPORTANT
# A map from probe type to geometry_parameters
# Here the key is "imDatPrb_type" in the metadata
# for a good understanding of this fields please read the _read_imro_string() function
# The main formula of the contact positions is this one
#     y_idx, x_idx = np.divmod(elec_ids, probe_description["ncols_per_shank"])
#    stagger = np.mod(y_idx + 1, 2) * probe_description["stagger"]
#    x_pos = x_idx * x_pitch + stagger
#    y_pos = y_idx * y_pitch

npx_descriptions = {
    # Neuropixels 1.0
    # This probably should be None or something else because NOT ONLY the neuropixels 1.0 have that imDatPrb_type
    "0": {
        "model_name": "Neuropixels 1.0",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 16,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 2.0 - Single Shank - Prototype
    "21": {
        "model_name": "Neuropixels 2.0 - Single Shank - Prototype",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Four Shank - Prototype
    "24": {
        "model_name": "Neuropixels 2.0 - Four Shank - Prototype",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Neuropixels 2.0 - Single Shank - Commercial without metal cap
    "2003": {
        "model_name": "Neuropixels 2.0 - Single Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Single Shank - Commercial with metal cap
    "2004": {
        "model_name": "Neuropixels 2.0 - Single Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Four Shank - Commercial without metal cap
    "2013": {
        "model_name": "Neuropixels 2.0 - Four Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Neuropixels 2.0 - Four Shank - Commercial with metal cap
    "2014": {
        "model_name": "Neuropixels 2.0 - Four Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Neuropixels 2.0 Quad Base
    "2020": {
        "model_name": "Neuropixels 2.0 - Quad Base",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Experimental probes previous to 1.0
    "Phase3a": {
        "model_name": "Phase3a",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 16.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
        ),
    },
    # Neuropixels 1.0-NHP Short (10mm)
    "1015": {
        "model_name": "Neuropixels 1.0-NHP - short",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    #################
    # Neuropixels 1.0-NHP Medium (25mm)
    "1020": {
        "model_name": "Neuropixels 1.0-NHP - medium - staggered",
        "x_pitch": 103 - 12,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 1248,  ### verify this number!!!!!!! Jennifer Colonell has 1368
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 1.0-NHP Medium (25mm)
    "1021": {
        "model_name": "Neuropixels 1.0-NHP - medium - staggered",
        "x_pitch": 103 - 12,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 1248,  ### verify this number!!!!!!! Jennifer Colonell has 1368
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    #################
    # Neuropixels 1.0-NHP Medium (25mm)
    "1022": {
        "model_name": "Neuropixels 1.0-NHP - medium",
        "x_pitch": 103,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 1248,  ### verify this number!!!!!!! Jennifer Colonell has 1368
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 1.0-NHP 45mm SOI90 - NHP long 90um wide, staggered contacts
    "1030": {
        "model_name": "Neuropixels 1.0-NHP - long SOI90 staggered",
        "x_pitch": 56,
        "y_pitch": 20,
        "stagger": 12,
        "contact_width": 12,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 2208,
        "contour_description": "nhp90",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 1.0-NHP 45mm SOI125 - NHP long 125um wide, staggered contacts
    "1031": {
        "model_name": "Neuropixels 1.0-NHP - long SOI125 staggered",
        "x_pitch": 91,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 2208,
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # 1.0-NHP 45mm SOI115 / 125 linear - NHP long 125um wide, linear contacts
    "1032": {
        "model_name": "Neuropixels 1.0-NHP - long SOI125 linear",
        "x_pitch": 103,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 2208,
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Ultra probes 1 bank
    "1100": {
        "model_name": "Neuropixels Ultra (1 bank)",
        "x_pitch": 6,
        "y_pitch": 6,
        "contact_width": 5,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 8,
        "nrows_per_shank": 48,
        "contour_description": "np70",
        "contour_shift": [-14, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Ultra probes 16 banks
    "1110": {
        "model_name": "Neuropixels Ultra (16 banks)",
        "x_pitch": 6,
        "y_pitch": 6,
        "contact_width": 5,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 8,
        "nrows_per_shank": 768,
        "contour_description": "np70",
        "contour_shift": [-14, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    "1121": {
        "model_name": "Neuropixels Ultra - Type 2",
        "x_pitch": 6,
        "y_pitch": 3,
        "contact_width": 2,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 1,
        "nrows_per_shank": 384,
        "contour_description": "np70",
        "contour_shift": [-6.25, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # NP-Opto
    "1300": {
        "model_name": "Neuropixels Opto",
        "x_pitch": 48,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
}


# Map imDatPrb_pn (probe number) to imDatPrb_type (probe type) when the latter is missing
probe_part_number_to_probe_type = {
    # for old version without a probe number we assume NP1.0
    None: "0",
    # NP1.0
    "PRB_1_4_0480_1": "0",
    "PRB_1_4_0480_1_C": "0",
    "PRB_1_2_0480_2": "0",
    "NP1010": "0",
    # NHP probes lin
    "NP1015": "1015",
    "NP1016": "1015",
    "NP1017": "1015",
    # NHP probes stag med
    "NP1020": "1020",
    "NP1021": "1021",
    "NP1022": "1022",
    # NHP probes stag long
    "NP1030": "1030",
    "NP1031": "1031",
    "NP1032": "1032",
    # NP2.0
    "NP2000": "21",
    "NP2010": "24",
    "NP2013": "2013",
    "NP2014": "2014",
    "NP2003": "2003",
    "NP2004": "2004",
    "PRB2_1_2_0640_0": "21",
    "PRB2_4_2_0640_0": "24",
    # NXT
    "NP2020": "2020",
    # Ultra
    "NP1100": "1100",  # Ultra probe - 1 bank
    "NP1110": "1110",  # Ultra probe - 16 banks no handle beacuse
    "NP1121": "1121",  # Ultra probe - beta configuration
    # Opto
    "NP1300": "1300",  # Opto probe
    # missing NP1200
    # missing NXT3000
}


def read_imro(file_path: Union[str, Path]) -> Probe:
    """
    Read probe position from the imro file used in input of SpikeGlx and Open-Ephys for neuropixels probes.

    Parameters
    ----------
    file_path : Path or str
        The .imro file path

    Returns
    -------
    probe : Probe object

    """
    # the input is an imro file
    meta_file = Path(file_path)
    assert meta_file.suffix == ".imro", "'file' should point to the .imro file"
    with meta_file.open(mode="r") as f:
        imro_str = str(f.read())
    return _read_imro_string(imro_str)


def _make_npx_probe_from_description(probe_description, elec_ids, shank_ids):
    # used by _read_imro_string and for generating the NP library

    model_name = probe_description["model_name"]

    # compute position
    y_idx, x_idx = np.divmod(elec_ids, probe_description["ncols_per_shank"])
    x_pitch = probe_description["x_pitch"]
    y_pitch = probe_description["y_pitch"]

    stagger = np.mod(y_idx + 1, 2) * probe_description["stagger"]
    x_pos = x_idx * x_pitch + stagger
    y_pos = y_idx * y_pitch

    # if probe_description["shank_number"] > 1:
    if shank_ids is not None:
        # shank_ids = np.array(contact_info["shank_id"])
        shank_pitch = probe_description["shank_pitch"]
        contact_ids = [f"s{shank_id}e{elec_id}" for shank_id, elec_id in zip(shank_ids, elec_ids)]
        x_pos += np.array(shank_ids).astype(int) * shank_pitch
    else:
        # shank_ids = None
        contact_ids = [f"e{elec_id}" for elec_id in elec_ids]

    positions = np.stack((x_pos, y_pos), axis=1)

    # construct Probe object
    probe = Probe(ndim=2, si_units="um", model_name=model_name, manufacturer="IMEC")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shank_ids=shank_ids,
        shape_params={"width": probe_description["contact_width"]},
    )

    probe.set_contact_ids(contact_ids)

    # Add planar contour
    contour_description = probe_description["contour_description"]
    polygon = np.array(polygon_contour_description[contour_description])

    contour = []
    shank_pitch = probe_description["shank_pitch"]
    for shank_id in range(probe_description["shank_number"]):
        shank_shift = np.array([shank_pitch * shank_id, 0])
        contour += list(polygon + shank_shift)

    # final contour_shift
    contour_shift = np.array(probe_description["contour_shift"])
    contour = np.array(contour) + contour_shift
    probe.set_planar_contour(contour)

    # shank tips : minimum of the polygon
    shank_tips = []
    for shank_id in range(probe_description["shank_number"]):
        shank_shift = np.array([shank_pitch * shank_id, 0])
        shank_tip = np.array(polygon[2]) + contour_shift + shank_shift
        shank_tips.append(shank_tip.tolist())

    probe.annotate(shank_tips=shank_tips)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def _read_imro_string(imro_str: str, imDatPrb_pn: Optional[str] = None) -> Probe:
    """
    Parse the IMRO table when presented as a string and create a Probe object.

    Parameters
    ----------
    imro_str : str
        IMRO table as a string.
    imDatPrb_pn : str, optional
        Probe number, by default None.

    Returns
    -------
    Probe
        A Probe object built from  the parsed IMRO table data.

    See Also
    --------
    https://billkarsh.github.io/SpikeGLX/help/imroTables/

    """
    imro_table_header_str, *imro_table_values_list, _ = imro_str.strip().split(")")
    imro_table_header = tuple(map(int, imro_table_header_str[1:].split(",")))

    if imDatPrb_pn is None:
        if len(imro_table_header) == 3:
            # In older versions of neuropixel arrays (phase 3A), imro tables were structured differently.
            probe_serial_number, probe_option, num_contact = imro_table_header
            imDatPrb_type = "Phase3a"
        elif len(imro_table_header) == 2:
            imDatPrb_type, num_contact = imro_table_header
        else:
            raise ValueError(f"read_imro error, the header has a strange length: {imro_table_header}")
        imDatPrb_type = str(imDatPrb_type)
    else:
        if imDatPrb_pn not in probe_part_number_to_probe_type:
            raise NotImplementedError(f"Probe part number {imDatPrb_pn} is not supported yet")
        imDatPrb_type = probe_part_number_to_probe_type[imDatPrb_pn]

    probe_description = npx_descriptions[imDatPrb_type]

    fields = probe_description["fields_in_imro_table"]
    contact_info = {k: [] for k in fields}
    for field_values_str in imro_table_values_list:  # Imro table values look like '(value, value, value, ... '
        values = tuple(map(int, field_values_str[1:].split(" ")))
        # Split them by space to get (int('value'), int('value'), int('value'), ...)
        for field, field_value in zip(fields, values):
            contact_info[field].append(field_value)

    channel_ids = np.array(contact_info["channel_ids"])
    if "elec_ids" in contact_info:
        elec_ids = np.array(contact_info["elec_ids"])
    else:
        banks = np.array(contact_info["banks"])
        elec_ids = banks * 384 + channel_ids

    if probe_description["shank_number"] > 1:
        shank_ids = np.array(contact_info["shank_id"])
        # shank_pitch = probe_description["shank_pitch"]
        # contact_ids = [f"s{shank_id}e{elec_id}" for shank_id, elec_id in zip(shank_ids, elec_ids)]
        # x_pos += np.array(shank_ids).astype(int) * shank_pitch
    else:
        shank_ids = None
        # contact_ids = [f"e{elec_id}" for elec_id in elec_ids]

    probe = _make_npx_probe_from_description(probe_description, elec_ids, shank_ids)

    # this is scalar annotations
    probe.annotate(
        probe_type=imDatPrb_type,
    )

    # this is vector annotations
    vector_properties = ("channel_ids", "banks", "references", "ap_gains", "lf_gains", "ap_hp_filters")
    vector_properties_available = {k: v for k, v in contact_info.items() if k in vector_properties}
    probe.annotate_contacts(**vector_properties_available)

    return probe


def write_imro(file: str | Path, probe: Probe):
    """
    save imro file (`.imrc`, imec readout) in a file.
    https://github.com/open-ephys-plugins/neuropixels-pxi/blob/master/Source/Formats/IMRO.h

    Parameters
    ----------
    file : Path or str
        The file path
    probe : Probe object

    """
    probe_type = probe.annotations["probe_type"]
    data = probe.to_dataframe(complete=True).sort_values("device_channel_indices")
    annotations = probe.contact_annotations
    ret = [f"({probe_type},{len(data)})"]

    if probe_type == "0":
        for ch in range(len(data)):
            ret.append(
                f"({ch} 0 {annotations['references'][ch]} {annotations['ap_gains'][ch]} "
                f"{annotations['lf_gains'][ch]} {annotations['ap_hp_filters'][ch]})"
            )

    elif probe_type in ("21", "2003", "2004"):
        for ch in range(len(data)):
            ret.append(
                f"({data['device_channel_indices'][ch]} {annotations['banks'][ch]} "
                f"{annotations['references'][ch]} {data['contact_ids'][ch][1:]})"
            )

    elif probe_type in ("24", "2013", "2014"):
        for ch in range(len(data)):
            ret.append(
                f"({data['device_channel_indices'][ch]} {data['shank_ids'][ch]} {annotations['banks'][ch]} "
                f"{annotations['references'][ch]} {data['contact_ids'][ch][3:]})"
            )
    else:
        raise ValueError(f"unknown imro type : {probe_type}")
    with open(file, "w") as f:
        f.write("".join(ret))


##
# SpikeGLX zone for neuropixel
##


def read_spikeglx(file: str | Path) -> Probe:
    """
    Read probe position for the meta file generated by SpikeGLX

    See http://billkarsh.github.io/SpikeGLX/#metadata-guides for implementation.
    The x_pitch/y_pitch/width are set automatically depending on the NP version.

    The shape is auto generated as a shank.

    Now reads:
      * NP0.0 (=phase3A)
      * NP1.0 (=phase3B2)
      * NP2.0 with 4 shank
      * NP1.0-NHP

    Parameters
    ----------
    file : Path or str
        The .meta file path

    Returns
    -------
    probe : Probe object

    """

    meta_file = Path(file)
    assert meta_file.suffix == ".meta", "'meta_file' should point to the .meta SpikeGLX file"

    meta = parse_spikeglx_meta(meta_file)

    assert "imroTbl" in meta, "Could not find imroTbl field in meta file!"
    imro_table = meta["imroTbl"]

    # read serial number
    imDatPrb_serial_number = meta.get("imDatPrb_sn", None)
    if imDatPrb_serial_number is None:  # this is for Phase3A
        imDatPrb_serial_number = meta.get("imProbeSN", None)

    # read other metadata
    imDatPrb_pn = meta.get("imDatPrb_pn", None)
    imDatPrb_port = meta.get("imDatPrb_port", None)
    imDatPrb_slot = meta.get("imDatPrb_slot", None)
    imDatPrb_part_number = meta.get("imDatPrb_pn", None)

    probe = _read_imro_string(imro_str=imro_table, imDatPrb_pn=imDatPrb_pn)

    # add serial number and other annotations
    probe.annotate(serial_number=imDatPrb_serial_number)
    probe.annotate(part_number=imDatPrb_part_number)
    probe.annotate(port=imDatPrb_port)
    probe.annotate(slot=imDatPrb_slot)
    probe.annotate(serial_number=imDatPrb_serial_number)

    # sometimes we need to slice the probe when not all channels are saved
    saved_chans = get_saved_channel_indices_from_spikeglx_meta(meta_file)
    # remove the SYS chans
    saved_chans = saved_chans[saved_chans < probe.get_contact_count()]
    if saved_chans.size != probe.get_contact_count():
        # slice if needed
        probe = probe.get_slice(saved_chans)
    # wire it
    probe.set_device_channel_indices(np.arange(probe.get_contact_count()))

    return probe


def parse_spikeglx_meta(meta_file: str | Path) -> dict:
    """
    Parse the "meta" file from spikeglx into a dict.
    All fields are kept in txt format and must also be parsed themselves.
    """
    meta_file = Path(meta_file)
    with meta_file.open(mode="r") as f:
        lines = f.read().splitlines()

    meta = {}
    for line in lines:
        split_str = line.split("=")
        key = split_str[0]
        val = "=".join(split_str[1:])
        if key.startswith("~"):
            key = key[1:]
        meta[key] = val

    return meta


def parse_spikeglx_snsGeomMap(meta):
    """
    For some meta recent file.
    There is a field 'snsGeomMap' that can be used for gettiing geometry of contacts.

    This is a good check that it corresponds to the 'imroTbl' for testing when present.

    Important notes :
      * the reference is not the same for x coordinates
      * x_pos, y_pos is referenced by shank, when using imro then x, y are absolute for the entire probe.
    """

    geom_list = meta["snsGeomMap"].split(sep=")")

    # first entry is for instance (NP1000,1,0,70)
    probe_type, num_shank, shank_pitch, shank_width = geom_list[0][1:].split(",")
    num_shank, shank_pitch, shank_width = int(num_shank), float(shank_pitch), float(shank_width)

    geom_list = geom_list[1:-1]
    num_contact = len(geom_list)

    shank_ids = np.zeros((num_contact,), dtype="int64")
    x_pos = np.zeros((num_contact,), "float64")
    y_pos = np.zeros((num_contact,), "float64")
    activated = np.zeros((num_contact,), "bool")

    # then it is instance a list like this (0:27:0:1)(0:59:0:1)(0:27:15:1)...
    for i in range(num_contact):
        shank_id, x, y, act = geom_list[i][1:].split(":")
        shank_ids[i] = int(shank_id)
        x_pos[i] = float(x)
        y_pos[i] = float(y)
        activated[i] = bool(act)

    return num_shank, shank_width, shank_pitch, shank_ids, x_pos, y_pos, activated


# def spikeglx_snsGeomMap_to_probe(meta):
#     parse_spikeglx_snsGeomMap(meta)


def get_saved_channel_indices_from_spikeglx_meta(meta_file: str | Path) -> np.array:
    """
    Utils function to get the saved channels.

    It uses the 'snsSaveChanSubset' field in  the meta file, which is as follows:
    snsSaveChanSubset=0:10,50:55,100
    with chan1:chan2 chan2 inclusive

    This function come from here Jennifer Colonell
    https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/common/SGLXMetaToCoords.py#L65

    """
    meta_file = Path(meta_file)
    meta = parse_spikeglx_meta(meta_file)
    chans_txt = meta["snsSaveChanSubset"]

    if chans_txt == "all":
        chans = np.arange(int(meta["nSavedChans"]))
    else:
        chans = []
        for e in chans_txt.split(","):
            if ":" in e:
                start, stop = e.split(":")
                start, stop = int(start), int(stop) + 1
                chans.extend(np.arange(start, stop))
            else:
                chans.append(int(e))
    chans = np.array(chans, dtype="int64")
    return chans


##
# OpenEphys zone for neuropixel
##


def read_openephys(
    settings_file: str | Path,
    stream_name: Optional[str] = None,
    probe_name: Optional[str] = None,
    serial_number: Optional[str] = None,
    fix_x_position_for_oe_5: bool = True,
    raise_error: bool = True,
) -> Probe:
    """
    Read probe positions from Open Ephys folder when using the Neuropix-PXI plugin.
    The reader assumes that the NP_PROBE fields are available in the settings file.
    Open Ephys versions 0.5.x and 0.6.x are supported:
    * For version 0.6.x, the probe names are inferred from the STREAM field. Probe
      information is then populated sequentially with the NP_PROBE fields.
    * For version 0.5.x, STREAMs are not available. In this case, if multiple probes
      are available, they are named sequentially based on the nodeId. E.g. "100.0",
      "100.1". These substrings are used for selection.

    Parameters
    ----------
    settings_file :  Path, str, or None
        If more than one settings.xml file is in the folder structure, this argument
        is required to indicate which settings file to use
    stream_name : str or None
        If more than one probe is used, the 'stream_name' indicates which probe to load base on the
        stream. For example, if there are 3 probes ('ProbeA', 'ProbeB', ProbeC) and the stream_name is
        contains the substring 'ProbeC' (e.g. 'my-stream-ProbeC'), then the probe associated with
        ProbeC is returned. If this argument is used, the 'probe_name' and 'serial_number' must be None.
    probe_name : str or None
        If more than one probe is used, the 'probe_name' indicates which probe to load base on the
        probe name (e.g. "ProbeB"). If this argument is used, the 'stream_name' and 'serial_number'
        must be None.
    serial_number : str or None
        If more than one probe is used, the 'serial_number' indicates which probe to load base on the
        serial number. If this argument is used, the 'stream_name' and 'probe_name'
        must be None.
    fix_x_position_for_oe_5: bool
        The neuropixels PXI plugin in the open-ephys < 0.6.0 contains a bug in the y position. This option allow to fix it.
    raise_error: bool
        If True, any error would raise an exception. If False, None is returned. Default True

    Note
    ----
    The electrode positions are only available when recording using the Neuropix-PXI plugin version >= 0.3.3

    Returns
    -------
    probe : Probe object

    """

    ET = import_safely("xml.etree.ElementTree")
    # parse xml
    tree = ET.parse(str(settings_file))
    root = tree.getroot()

    info_chain = root.find("INFO")
    oe_version = parse(info_chain.find("VERSION").text)
    neuropix_pxi_processor = None
    onebox_processor = None
    for signal_chain in root.findall("SIGNALCHAIN"):
        for processor in signal_chain:
            if "PROCESSOR" == processor.tag:
                name = processor.attrib["name"]
                if "Neuropix-PXI" in name:
                    neuropix_pxi_processor = processor
                if "OneBox" in name:
                    onebox_processor = processor

    if neuropix_pxi_processor is None and onebox_processor is None:
        if raise_error:
            raise Exception("Open Ephys can only be read when the Neuropix-PXI or the " "OneBox plugin is used.")
        return None

    if neuropix_pxi_processor is not None:
        assert onebox_processor is None, "Only one processor should be present"
        processor = neuropix_pxi_processor
        neuropix_pxi_version = parse(neuropix_pxi_processor.attrib["libraryVersion"])
        if neuropix_pxi_version < parse("0.3.3"):
            if raise_error:
                raise Exception("Electrode locations are available from Neuropix-PXI version 0.3.3")
            return None
    if onebox_processor is not None:
        assert neuropix_pxi_processor is None, "Only one processor should be present"
        processor = onebox_processor

    if "NodeId" in processor.attrib:
        node_id = processor.attrib["NodeId"]
    elif "nodeId" in processor.attrib:
        node_id = processor.attrib["nodeId"]
    else:
        node_id = None

    # read STREAM fields if present (>=0.6.x)
    stream_fields = processor.findall("STREAM")
    if len(stream_fields) > 0:
        has_streams = True
        streams = []
        # find probe names (exclude ADC streams)
        for stream_field in stream_fields:
            stream = stream_field.attrib["name"]
            if "ADC" not in stream:
                streams.append(stream)
        # find probe names (exclude ADC streams)
        probe_names_used = np.unique([stream.split("-")[0] for stream in streams])
    else:
        has_streams = False
        probe_names_used = None

    # for Open Ephys version < 1.0 np_probes is in the EDITOR field.
    # for Open Ephys version >= 1.0 np_probes is in the CUSTOM_PARAMETERS field.
    editor = processor.find("EDITOR")
    if oe_version < parse("0.9.0"):
        np_probes = editor.findall("NP_PROBE")
    else:
        custom_parameters = editor.find("CUSTOM_PARAMETERS")
        np_probes = custom_parameters.findall("NP_PROBE")

    if len(np_probes) == 0:
        if raise_error:
            raise Exception("NP_PROBE field not found in settings")
        return None

    # In neuropixel plugin 0.7.0, the option for enabling/disabling probes was added.
    # Make sure we only keep enabled probes.
    np_probes = [probe for probe in np_probes if probe.attrib.get("isEnabled", "1") == "1"]
    if len(np_probes) == 0:
        if raise_error:
            raise Exception("No enabled probes found in settings")
        return None

    # read probes info
    # If STREAMs are not available, probes are sequentially named based on the node id
    if not has_streams:
        probe_names_used = [f"{node_id}.{stream_index}" for stream_index in range(len(np_probes))]

    # check consistency with stream names and other fields
    if has_streams:
        # make sure we have at least as many NP_PROBE as the number of used probes
        if len(np_probes) < len(probe_names_used):
            if raise_error:
                raise Exception(
                    f"Not enough NP_PROBE entries ({len(np_probes)}) " f"for used probes: {probe_names_used}"
                )
            return None

    # now load probe info from NP_PROBE fields
    np_probes_info = []
    for probe_idx, np_probe in enumerate(np_probes):
        slot = np_probe.attrib["slot"]
        port = np_probe.attrib["port"]
        dock = np_probe.attrib["dock"]
        probe_part_number = np_probe.attrib["probe_part_number"]
        probe_serial_number = np_probe.attrib["probe_serial_number"]
        # read channels
        channels = np_probe.find("CHANNELS")
        channel_names = np.array(list(channels.attrib.keys()))
        channel_ids = np.array([int(ch[2:]) for ch in channel_names])
        channel_order = np.argsort(channel_ids)

        # sort channel_names and channel_values
        channel_names = channel_names[channel_order]
        channel_values = np.array(list(channels.attrib.values()))[channel_order]

        # check if shank ids is present
        if all(":" in val for val in channel_values):
            shank_ids = np.array([int(val.split(":")[1]) for val in channel_values])
        elif all("_" in val for val in channel_names):
            shank_ids = np.array([int(val.split("_")[1]) for val in channel_names])
        else:
            shank_ids = None

        electrode_xpos = np_probe.find("ELECTRODE_XPOS")
        electrode_ypos = np_probe.find("ELECTRODE_YPOS")

        if electrode_xpos is None or electrode_ypos is None:
            if raise_error:
                raise Exception("ELECTRODE_XPOS or ELECTRODE_YPOS is not available in settings!")
            return None
        xpos = np.array([float(electrode_xpos.attrib[ch]) for ch in channel_names])
        ypos = np.array([float(electrode_ypos.attrib[ch]) for ch in channel_names])
        positions = np.array([xpos, ypos]).T

        probe_part_number = np_probe.get("probe_part_number", None)
        if probe_part_number not in probe_part_number_to_probe_type:
            raise NotImplementedError(f"Probe part number {probe_part_number} is not supported yet")
        ptype = probe_part_number_to_probe_type[probe_part_number]

        if fix_x_position_for_oe_5 and oe_version < parse("0.6.0") and shank_ids is not None:
            positions[:, 1] = positions[:, 1] - npx_descriptions[ptype]["shank_pitch"] * shank_ids

        # x offset so that the first column is at 0x
        offset = np.min(positions[:, 0])
        # if some shanks are not used, we need to adjust the offset
        if shank_ids is not None:
            offset -= np.min(shank_ids) * npx_descriptions[ptype]["shank_pitch"]
        positions[:, 0] -= offset

        contact_ids = []
        for i, pos in enumerate(positions):
            if ptype is None:
                contact_ids = None
                break

            stagger = np.mod(pos[1] / npx_descriptions[ptype]["y_pitch"] + 1, 2) * npx_descriptions[ptype]["stagger"]
            shank_id = shank_ids[i] if npx_descriptions[ptype]["shank_number"] > 1 else 0

            contact_id = int(
                (pos[0] - stagger - npx_descriptions[ptype]["shank_pitch"] * shank_id)
                / npx_descriptions[ptype]["x_pitch"]
                + npx_descriptions[ptype]["ncols_per_shank"] * pos[1] / npx_descriptions[ptype]["y_pitch"]
            )
            if npx_descriptions[ptype]["shank_number"] > 1:
                contact_ids.append(f"s{shank_id}e{contact_id}")
            else:
                contact_ids.append(f"e{contact_id}")

        model_name = npx_descriptions[ptype]["model_name"] if ptype is not None else "Unknown"
        np_probe_dict = {
            "model_name": model_name,
            "shank_ids": shank_ids,
            "contact_ids": contact_ids,
            "positions": positions,
            "slot": slot,
            "port": port,
            "dock": dock,
            "serial_number": probe_serial_number,
            "part_number": probe_part_number,
            "ptype": ptype,
        }
        # Sequentially assign probe names
        if "custom_probe_name" in np_probe.attrib and np_probe.attrib["custom_probe_name"] != probe_serial_number:
            name = np_probe.attrib["custom_probe_name"]
        else:
            name = probe_names_used[probe_idx]
        np_probe_dict.update({"name": name})
        np_probes_info.append(np_probe_dict)

    # now select correct probe (if multiple)
    if len(np_probes) > 1:
        found = False
        probe_names = [p["name"] for p in np_probes_info]

        if stream_name is not None:
            assert probe_name is None and serial_number is None, (
                "Use one of 'stream_name', 'probe_name', " "or 'serial_number'"
            )
            for probe_idx, probe_info in enumerate(np_probes_info):
                if probe_info["name"] in stream_name or probe_info["serial_number"] in stream_name:
                    found = True
                    break
            if not found:
                if raise_error:
                    raise Exception(
                        f"The stream {stream_name} is not associated to an available probe: {probe_names_used}"
                    )
                return None
        elif probe_name is not None:
            assert stream_name is None and serial_number is None, (
                "Use one of 'stream_name', 'probe_name', " "or 'serial_number'"
            )
            for probe_idx, probe_info in enumerate(np_probes_info):
                if probe_info["name"] == probe_name:
                    found = True
                    break
            if not found:
                if raise_error:
                    raise Exception(f"The provided {probe_name} is not in the available probes: {probe_names_used}")
                return None
        elif serial_number is not None:
            assert stream_name is None and probe_name is None, (
                "Use one of 'stream_name', 'probe_name', " "or 'serial_number'"
            )
            for probe_idx, probe_info in enumerate(np_probes_info):
                if probe_info["serial_number"] == str(serial_number):
                    found = True
                    break
            if not found:
                np_serial_numbers = [p["serial_number"] for p in probe_info]
                if raise_error:
                    raise Exception(
                        f"The provided {serial_number} is not in the available serial numbers: {np_serial_numbers}"
                    )
                return None
        else:
            raise Exception(
                f"More than one probe found. Use one of 'stream_name', 'probe_name', or 'serial_number' "
                f"to select the right probe.\nProbe names: {probe_names}"
            )
    else:
        # in case of a single probe, make sure it is consistent with optional
        # stream_name, probe_name, or serial number
        available_probe_name = np_probes_info[0]["name"]
        available_serial_number = np_probes_info[0]["serial_number"]

        if stream_name:
            if available_probe_name not in stream_name:
                if raise_error:
                    raise Exception(
                        f"Inconsistency between provided stream {stream_name} and available probe "
                        f"{available_probe_name}"
                    )
                return None
        if probe_name:
            if probe_name != available_probe_name:
                if raise_error:
                    raise Exception(
                        f"Inconsistency between provided probe name {probe_name} and available probe "
                        f"{available_probe_name}"
                    )
                return None
        if serial_number:
            if str(serial_number) != available_serial_number:
                if raise_error:
                    raise Exception(
                        f"Inconsistency between provided serial number {serial_number} and available serial numbers "
                        f"{available_serial_number}"
                    )
                return None
        probe_idx = 0

    np_probe_info = np_probes_info[probe_idx]
    np_probe = np_probes[probe_idx]
    positions = np_probe_info["positions"]
    shank_ids = np_probe_info["shank_ids"]

    ptype = np_probe_info["ptype"]
    if ptype in npx_descriptions:
        contact_width = npx_descriptions[ptype]["contact_width"]
        shank_pitch = npx_descriptions[ptype]["shank_pitch"]
        num_shanks = npx_descriptions[ptype]["shank_number"]
        contour_description = npx_descriptions[ptype]["contour_description"]
    else:
        contact_width = 12
        shank_pitch = 250
        num_shanks = 1
        contour_description = npx_descriptions["0"]["contour_description"]

    contact_ids = np_probe_info["contact_ids"] if np_probe_info["contact_ids"] is not None else None

    # check if subset of channels
    chans_saved = get_saved_channel_indices_from_openephys_settings(settings_file, stream_name=stream_name)

    # if a recording state is found, slice probe
    if chans_saved is not None:
        positions = positions[chans_saved]
        if shank_ids is not None:
            shank_ids = np.array(shank_ids)[chans_saved]
        if contact_ids is not None:
            contact_ids = np.array(contact_ids)[chans_saved]

    probe = Probe(
        ndim=2,
        si_units="um",
        name=np_probe_info["name"],
        serial_number=np_probe_info["serial_number"],
        manufacturer="IMEC",
        model_name=np_probe_info["model_name"],
    )
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shank_ids=shank_ids,
        shape_params={"width": contact_width},
    )
    probe.annotate(
        part_number=np_probe_info["part_number"],
        slot=np_probe_info["slot"],
        dock=np_probe_info["dock"],
        port=np_probe_info["port"],
    )

    if contact_ids is not None:
        probe.set_contact_ids(contact_ids)

    polygon = polygon_contour_description[contour_description]
    contour_shift = np.array(npx_descriptions[ptype]["contour_shift"])
    if shank_ids is None:
        contour = polygon
    else:
        contour = []
        for i in range(num_shanks):
            contour += list(np.array(polygon) + [shank_pitch * i, 0])

    # shift
    contour = np.array(contour) + contour_shift
    probe.set_planar_contour(contour)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def get_saved_channel_indices_from_openephys_settings(
    settings_file: str | Path, stream_name: str
) -> Optional[np.array]:
    """
    Returns an array with the subset of saved channels indices (if used)

    Parameters
    ----------
    settings_file : str or Path
        The path to the settings file
    stream_name : str
        The stream name that contains the probe name
        For example, "Record Node 100#ProbeA-AP" will select the AP stream of ProbeA.

    Returns
    -------
    chans_saved
        np.array of saved channel indices or None
    """
    # check if subset of channels
    ET = import_safely("xml.etree.ElementTree")
    # parse xml
    tree = ET.parse(str(settings_file))
    root = tree.getroot()

    signal_chain = root.find("SIGNALCHAIN")
    record_node = None
    for processor in signal_chain:
        if "PROCESSOR" == processor.tag:
            name = processor.attrib["name"]
            if "Record Node" in name:
                record_node = processor
                break
    chans_saved = None
    if record_node is not None:
        custom_params = record_node.find("CUSTOM_PARAMETERS")
        if custom_params is not None:
            custom_streams = custom_params.findall("STREAM")
            custom_stream_names = [stream.attrib["name"] for stream in custom_streams]
            if len(custom_streams) > 0:
                recording_states = [stream.attrib.get("recording_state", None) for stream in custom_streams]
                has_custom_states = False
                for rs in recording_states:
                    if rs is not None and rs not in ("ALL", "NONE"):
                        has_custom_states = True
                if has_custom_states:
                    if len(custom_streams) > 1:
                        assert stream_name is not None, (
                            f"More than one stream found with custom parameters: {custom_stream_names}. "
                            f"Use the `stream_name` argument to choose the correct stream"
                        )
                        possible_custom_streams = [
                            stream for stream in custom_streams if stream.attrib["name"] in stream_name
                        ]
                        if len(possible_custom_streams) > 1:
                            warnings.warn(
                                f"More than one custom parameters associated to {stream_name} "
                                f"found. Using fisrt one"
                            )
                        custom_stream = possible_custom_streams[0]
                    else:
                        custom_stream = custom_streams[0]
                    recording_state = custom_stream.attrib.get("recording_state", None)
                    if recording_state is not None:
                        if recording_state not in ("ALL", "NONE"):
                            chans_saved = np.array([chan for chan, r in enumerate(recording_state) if int(r) == 1])
    return chans_saved
