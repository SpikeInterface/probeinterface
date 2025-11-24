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
import json
import numpy as np

from .probe import Probe
from .utils import import_safely


global _np_probe_features
_np_probe_features = None


def _load_np_probe_features():
    # this avoid loading the json several times
    global _np_probe_features
    if _np_probe_features is None:
        probe_features_filepath = Path(__file__).absolute().parent / Path("resources/neuropixels_probe_features.json")
        _np_probe_features = json.load(open(probe_features_filepath, "r"))
    return _np_probe_features


# Map imDatPrb_pn (probe number) to imDatPrb_type (probe type) when the latter is missing
# ONLY needed for `read_imro` function
probe_part_number_to_probe_type = {
    # for old version without a probe number we assume NP1.0
    None: "0",
    # NP1.0
    "PRB_1_4_0480_1": "0",
    "PRB_1_4_0480_1_C": "0",  # This is the metal cap version
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
    "NP1110": "1110",  # Ultra probe - 16 banks no handle because
    "NP1121": "1121",  # Ultra probe - beta configuration
    # Opto
    "NP1300": "1300",  # Opto probe
}

# Map from imro format to ProbeInterface naming conventions
imro_field_to_pi_field = {
    "ap_gain": "ap_gains",
    "ap_hipas_flt": "ap_hp_filters",
    "bank": "banks",
    "bank_mask": "banks",
    "channel": "channel_ids",
    "electrode": "elec_ids",
    "lf_gain": "lf_gains",
    "ref_id": "references",
    "shank": "shank_id",
    "group": "group",
    "bankA": "bankA",
    "bankB": "bankB",
}


def get_probe_length(probe_part_number: str) -> int:
    """
    Returns the length of a given probe. We assume a length of
    1cm (10_000 microns) by default.

    Parameters
    ----------
    probe_part_number : str
        The part number of the probe e.g. 'NP2013'.

    Returns
    -------
    probe_length : int
        Length of full probe (microns)
    """

    probe_length = 10_000

    return probe_length


def make_mux_table_array(mux_information) -> np.array:
    """
    Function to parse the mux_table from ProbeTable.

    Parameters
    ----------
    mux_information : str
        The information from `z_mux_tables` in the ProbeTable `probe_feature.json` file

    Returns
    -------
    adc_groups_array : np.array
        Array of which channels are in each adc group, shaped (number of `adc`s, number of channels in each `adc`).
    """

    # mux_information looks like (num_adcs num_channels_per_adc)(int int int ...)(int int int ...)...(int int int ...)
    # First split on ')(' to get a list of the information in the brackets, and remove the leading data
    adc_info = mux_information.split(")(")[0]
    split_mux = mux_information.split(")(")[1:]

    # The first element is the number of ADCs and the number of channels per ADC
    num_adcs, num_channels_per_adc = map(int, adc_info[1:].split(","))

    # Then remove the brackets, and split using " " to get each integer as a list
    adc_groups = [
        np.array(each_mux.replace("(", "").replace(")", "").split(" ")).astype("int") for each_mux in split_mux
    ]
    adc_groups_array = np.transpose(np.array(adc_groups))

    return num_adcs, num_channels_per_adc, adc_groups_array


def get_probe_contour_vertices(shank_width, tip_length, probe_length) -> list:
    """
    Function to get the vertices of the probe contour from probe properties.
    The probe contour can be constructed from five points.

    These are the vertices shown in the following figure:

            Top of probe (y = probe_length)
        A +-------------------------------+ E
          |                               |
          |                               |
          |        Shank body             |
          |      (shank_width)            |
          |                               |
          |                               |
        B +-------------------------------+ D  (y = 0)
           \\                             /
            \\         Tip region        /
             \\      (tip_length)       /
              \\                       /
               \\                     /
                \\                   /
                 +-----------------+ C  (y = -tip_length)

    This function returns the vertices in the order [A, B, C, D, E] as a list of (x, y) coordinates.

    Parameters
    ----------
    shank_width : float
        Width of shank (um).
    tip_length : float
        Length of tip of probe (um).
    probe_length : float
        Length of entire probe (um).

    Returns
    -------
    polygon_vertices : tuple of tuple
        Five vertices as (x, y) coordinate pairs in micrometers, returned in the
        order [A, B, C, D, E] corresponding to the diagram above:
        A = (0, probe_length) - top-left corner
        B = (0, 0) - bottom-left corner at shank base
        C = (shank_width/2, -tip_length) - tip point (center bottom)
        D = (shank_width, 0) - bottom-right corner at shank base
        E = (shank_width, probe_length) - top-right corner
    """

    # this dict define the contour for one shank (duplicated when several shanks so)
    # note that a final "contour_shift" is applied
    polygon_vertices = (
        (0, probe_length),
        (0, 0),
        (shank_width / 2, -tip_length),
        (shank_width, 0),
        (shank_width, probe_length),
    )

    return polygon_vertices


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

    imro_table_header_str, *imro_table_values_list, _ = imro_str.strip().split(")")
    imro_table_header = tuple(map(int, imro_table_header_str[1:].split(",")))

    if len(imro_table_header) == 3:
        # In older versions of neuropixel arrays (phase 3A), imro tables were structured differently.
        # We use probe_type "0", which maps to probe_part_number NP1010 as a proxy for Phase3a.
        imDatPrb_type = "0"
    elif len(imro_table_header) == 2:
        imDatPrb_type, _ = imro_table_header
    else:
        raise ValueError(f"read_imro error, the header has a strange length: {imro_table_header}")
    imDatPrb_type = str(imDatPrb_type)

    for probe_part_number, probe_type in probe_part_number_to_probe_type.items():
        if imDatPrb_type == probe_type:
            imDatPrb_pn = probe_part_number

    return _read_imro_string(imro_str, imDatPrb_pn)


def _make_npx_probe_from_description(probe_description, model_name, elec_ids, shank_ids, mux_info=None) -> Probe:
    # used by _read_imro_string and for generating the NP library

    # compute position
    y_idx, x_idx = np.divmod(elec_ids, probe_description["cols_per_shank"])
    x_pitch = probe_description["electrode_pitch_horz_um"]
    y_pitch = probe_description["electrode_pitch_vert_um"]

    raw_stagger = (
        probe_description["even_row_horz_offset_left_edge_to_leftmost_electrode_center_um"]
        - probe_description["odd_row_horz_offset_left_edge_to_leftmost_electrode_center_um"]
    )

    stagger = np.mod(y_idx + 1, 2) * raw_stagger
    x_pos = (x_idx * x_pitch + stagger).astype("float64")
    y_pos = (y_idx * y_pitch).astype("float64")

    # if probe_description["shank_number"] > 1:
    if shank_ids is not None:
        # shank_ids = np.array(contact_info["shank_id"])
        shank_pitch = probe_description["shank_pitch_um"]
        contact_ids = [f"s{shank_id}e{elec_id}" for shank_id, elec_id in zip(shank_ids, elec_ids)]
        x_pos += np.array(shank_ids).astype(int) * shank_pitch
    else:
        # shank_ids = None
        contact_ids = [f"e{elec_id}" for elec_id in elec_ids]

    positions = np.stack((x_pos, y_pos), axis=1)

    # construct Probe object
    probe = Probe(ndim=2, si_units="um", model_name=model_name, manufacturer="imec")
    probe.description = probe_description["description"]
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shank_ids=shank_ids,
        shape_params={"width": probe_description["electrode_size_horz_direction_um"]},
    )

    probe.set_contact_ids(contact_ids)

    # Add planar contour
    polygon = np.array(
        get_probe_contour_vertices(
            probe_description["shank_width_um"], probe_description["tip_length_um"], get_probe_length(model_name)
        )
    )

    contour = []
    shank_pitch = probe_description["shank_pitch_um"]
    for shank_id in range(probe_description["num_shanks"]):
        shank_shift = np.array([shank_pitch * shank_id, 0])
        contour += list(polygon + shank_shift)

    # final contour_shift
    middle_of_bottommost_electrode_to_top_of_shank_tip = 11
    contour_shift = np.array(
        [
            -probe_description["odd_row_horz_offset_left_edge_to_leftmost_electrode_center_um"],
            -middle_of_bottommost_electrode_to_top_of_shank_tip,
        ]
    )
    contour = np.array(contour) + contour_shift
    probe.set_planar_contour(contour)

    # shank tips : minimum of the polygon
    shank_tips = []
    for shank_id in range(probe_description["num_shanks"]):
        shank_shift = np.array([shank_pitch * shank_id, 0])
        shank_tip = np.array(polygon[2]) + contour_shift + shank_shift
        shank_tips.append(shank_tip.tolist())

    probe.annotate(shank_tips=shank_tips)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    # set other key metadata annotations
    probe.annotate(
        adc_bit_depth=int(probe_description["adc_bit_depth"]),
        num_readout_channels=int(probe_description["num_readout_channels"]),
        ap_sample_frequency_hz=float(probe_description["ap_sample_frequency_hz"]),
        lf_sample_frequency_hz=float(probe_description["lf_sample_frequency_hz"]),
    )

    # annotate with MUX table
    if mux_info is not None:
        # annotate each contact with its mux channel
        num_adcs, num_channels_per_adc, mux_table = make_mux_table_array(mux_info)
        num_contacts = positions.shape[0]
        # ADC group: which adc is used for each contact
        adc_groups = np.zeros(num_contacts, dtype="int64")
        # ADC sample order: order of sampling of the contact in the adc group
        adc_sample_order = np.zeros(num_contacts, dtype="int64")
        for adc_idx, adc_groups_per_adc in enumerate(mux_table):
            adc_groups_per_adc = adc_groups_per_adc[adc_groups_per_adc < num_contacts]
            adc_groups[adc_groups_per_adc] = adc_idx
            adc_sample_order[adc_groups_per_adc] = np.arange(len(adc_groups_per_adc))
        probe.annotate(num_adcs=num_adcs)
        probe.annotate(num_channels_per_adc=num_channels_per_adc)
        probe.annotate_contacts(adc_group=adc_groups)
        probe.annotate_contacts(adc_sample_order=adc_sample_order)

    return probe


def build_neuropixels_probe(probe_part_number: str) -> Probe:
    """
    Build a Neuropixels probe with all possible contacts from the probe part number.

    This function constructs a complete probe geometry based on IMEC manufacturer specifications
    sourced from Bill Karsh's ProbeTable repository (https://github.com/billkarsh/ProbeTable).
    The specifications include contact positions, electrode dimensions, shank geometry, MUX routing
    tables, and ADC configurations. The resulting probe contains ALL electrodes (e.g., 960 for
    NP1.0, 1280 for NP2.0), not just the subset that might be recorded in an actual experiment.

    Parameters
    ----------
    probe_part_number : str
        Probe part number (specific SKU identifier).
        Examples: "NP1000", "NP2000", "NP1010", "NP2003", "NP2004"

        Note: This is the specific SKU, not the model/platform name:
        - probe_part_number is like "NP1000" (specific SKU)
        - NOT like "Neuropixels 1.0" or "Neuropixels 2.0" (platform family)

        In SpikeGLX meta files, this corresponds to the `imDatPrb_pn` field.
        Multiple part numbers may belong to the same platform family but have
        different configurations or variants.

    Returns
    -------
    probe : Probe
        The complete Probe object with all contacts and metadata.
    """
    # ===== 1. Load configuration =====
    probe_features = _load_np_probe_features()
    probe_spec_dict = probe_features["neuropixels_probes"][probe_part_number]

    # ===== 2. Calculate electrode IDs and shank IDs =====
    num_shanks = int(probe_spec_dict["num_shanks"])
    contacts_per_shank = int(probe_spec_dict["cols_per_shank"]) * int(probe_spec_dict["rows_per_shank"])

    if num_shanks == 1:
        elec_ids = np.arange(contacts_per_shank, dtype=int)
        shank_ids = None
    else:
        elec_ids = np.concatenate([np.arange(contacts_per_shank, dtype=int) for _ in range(num_shanks)])
        shank_ids = np.concatenate([np.zeros(contacts_per_shank, dtype=int) + i for i in range(num_shanks)])

    # ===== 3. Calculate contact positions =====
    cols_per_shank = int(probe_spec_dict["cols_per_shank"])
    y_idx, x_idx = np.divmod(elec_ids, cols_per_shank)

    x_pitch = float(probe_spec_dict["electrode_pitch_horz_um"])
    y_pitch = float(probe_spec_dict["electrode_pitch_vert_um"])

    even_offset = float(probe_spec_dict["even_row_horz_offset_left_edge_to_leftmost_electrode_center_um"])
    odd_offset = float(probe_spec_dict["odd_row_horz_offset_left_edge_to_leftmost_electrode_center_um"])
    raw_stagger = even_offset - odd_offset

    stagger = np.mod(y_idx + 1, 2) * raw_stagger
    x_pos = (x_idx * x_pitch + stagger).astype("float64")
    y_pos = (y_idx * y_pitch).astype("float64")

    # Apply horizontal offset for multi-shank probes
    if shank_ids is not None:
        shank_pitch = float(probe_spec_dict["shank_pitch_um"])
        x_pos += np.array(shank_ids).astype(int) * shank_pitch

    positions = np.stack((x_pos, y_pos), axis=1)

    # ===== 4. Calculate contact IDs =====
    shank_ids_iter = shank_ids if shank_ids is not None else [None] * len(elec_ids)
    contact_ids = [
        _build_canonical_contact_id(elec_id, shank_id) for shank_id, elec_id in zip(shank_ids_iter, elec_ids)
    ]

    # ===== 5. Create Probe object and set contacts =====
    probe = Probe(ndim=2, si_units="um", model_name=probe_part_number, manufacturer="imec")
    probe.description = probe_spec_dict["description"]
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shank_ids=shank_ids,
        shape_params={"width": float(probe_spec_dict["electrode_size_horz_direction_um"])},
    )
    probe.set_contact_ids(contact_ids)

    # ===== 6. Build probe contour and shank tips =====
    shank_width = float(probe_spec_dict["shank_width_um"])
    tip_length = float(probe_spec_dict["tip_length_um"])
    polygon = np.array(get_probe_contour_vertices(shank_width, tip_length, get_probe_length(probe_part_number)))

    # Build contour for all shanks
    contour = []
    if shank_ids is not None:
        shank_pitch = float(probe_spec_dict["shank_pitch_um"])
    for shank_id in range(num_shanks):
        shank_shift = np.array([shank_pitch * shank_id if shank_ids is not None else 0, 0])
        contour += list(polygon + shank_shift)

    # Apply contour shift to align with contact positions
    # This constant (11 μm) represents the vertical distance from the center of the bottommost
    # electrode to the top of the shank tip. This is a geometric constant for Neuropixels probes
    # that is not currently available in the ProbeTable specifications.
    middle_of_bottommost_electrode_to_top_of_shank_tip = 11
    contour_shift = np.array([-odd_offset, -middle_of_bottommost_electrode_to_top_of_shank_tip])
    contour = np.array(contour) + contour_shift
    probe.set_planar_contour(contour)

    # Calculate shank tips (polygon[2] is the tip vertex from get_probe_contour_vertices)
    tip_vertex = polygon[2]
    shank_tips = []
    for shank_id in range(num_shanks):
        shank_shift = np.array([shank_pitch * shank_id if shank_ids is not None else 0, 0])
        shank_tip = np.array(tip_vertex) + contour_shift + shank_shift
        shank_tips.append(shank_tip.tolist())
    probe.annotate(shank_tips=shank_tips)

    # ===== 7. Add metadata annotations =====
    probe.annotate(
        adc_bit_depth=int(probe_spec_dict["adc_bit_depth"]),
        num_readout_channels=int(probe_spec_dict["num_readout_channels"]),
        ap_sample_frequency_hz=float(probe_spec_dict["ap_sample_frequency_hz"]),
        lf_sample_frequency_hz=float(probe_spec_dict["lf_sample_frequency_hz"]),
    )

    # ===== 8. Add MUX table annotations =====
    mux_table_string = probe_features["z_mux_tables"][probe_spec_dict["mux_table_format_type"]]
    if mux_table_string is not None:
        # Parse MUX table string: (num_adcs,num_channels_per_adc)(int int ...)(int int ...)...
        adc_info = mux_table_string.split(")(")[0]
        split_mux = mux_table_string.split(")(")[1:]
        num_adcs, num_channels_per_adc = map(int, adc_info[1:].split(","))
        adc_groups_list = [
            np.array(each_mux.replace("(", "").replace(")", "").split(" ")).astype("int") for each_mux in split_mux
        ]
        mux_table = np.transpose(np.array(adc_groups_list))

        # Map contacts to ADC groups and sample order
        num_contacts = positions.shape[0]
        adc_groups = np.zeros(num_contacts, dtype="int64")
        adc_sample_order = np.zeros(num_contacts, dtype="int64")
        for adc_index, adc_groups_per_adc in enumerate(mux_table):
            adc_groups_per_adc = adc_groups_per_adc[adc_groups_per_adc < num_contacts]
            adc_groups[adc_groups_per_adc] = adc_index
            adc_sample_order[adc_groups_per_adc] = np.arange(len(adc_groups_per_adc))

        probe.annotate(num_adcs=num_adcs)
        probe.annotate(num_channels_per_adc=num_channels_per_adc)
        probe.annotate_contacts(adc_group=adc_groups)
        probe.annotate_contacts(adc_sample_order=adc_sample_order)

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

    probe_type_num_chans, *imro_table_values_list, _ = imro_str.strip().split(")")

    # probe_type_num_chans looks like f"({probe_type},{num_chans}"
    probe_type = probe_type_num_chans.split(",")[0][1:]

    probe_features = _load_np_probe_features()
    pt_metadata, fields, mux_info = get_probe_metadata_from_probe_features(probe_features, imDatPrb_pn)

    # fields = probe_description["fields_in_imro_table"]
    contact_info = {k: [] for k in fields}
    for field_values_str in imro_table_values_list:  # Imro table values look like '(value, value, value, ... '
        # Split them by space to get int('value'), int('value'), int('value'), ...)
        values = tuple(map(int, field_values_str[1:].split(" ")))
        for field, field_value in zip(fields, values):
            contact_info[field].append(field_value)

    channel_ids = np.array(contact_info["channel"])
    if "electrode" in contact_info:
        elec_ids = np.array(contact_info["electrode"])
    else:
        if contact_info.get("bank") is not None:
            bank_key = "bank"
        elif contact_info.get("bank_mask") is not None:
            bank_key = "bank_mask"
        banks = np.array(contact_info[bank_key])
        elec_ids = banks * 384 + channel_ids

    if pt_metadata["num_shanks"] > 1:
        shank_ids = np.array(contact_info["shank"])
    else:
        shank_ids = None

    probe = _make_npx_probe_from_description(pt_metadata, imDatPrb_pn, elec_ids, shank_ids, mux_info)

    # scalar annotations
    probe.annotate(
        probe_type=probe_type,
    )

    # vector annotations
    vector_properties = ("channel", "bank", "bank_mask", "ref_id", "ap_gain", "lf_gain", "ap_hipas_flt")

    vector_properties_available = {}
    for k, v in contact_info.items():
        if (k in vector_properties) and (len(v) > 0):
            # convert to ProbeInterface naming for backwards compatibility
            vector_properties_available[imro_field_to_pi_field.get(k)] = v

    probe.annotate_contacts(**vector_properties_available)

    return probe


def get_probe_metadata_from_probe_features(probe_features: dict, imDatPrb_pn: str):
    """
    Parses the `probe_features` dict, to cast string to appropriate types
    and parses the imro_table_fields string. Returns the metadata needed
    to construct a probe with part number `imDatPrb_pn`.

    Parameters
    ----------
    probe_features : dict
        Dictionary obtained when reading in the `neuropixels_probe_features.json` file.
    imDatPrb_pn : str
       Probe part number.

    Returns
    -------
    probe_metadata, imro_field, mux_information
        Dictionary of probe metadata.
        Tuple of fields included in the `imro_table_fields`.
        Mux table information, if available, as a string.
    """

    probe_metadata = probe_features["neuropixels_probes"].get(imDatPrb_pn)
    for key in probe_metadata.keys():
        if key in ["num_shanks", "cols_per_shank", "rows_per_shank", "adc_bit_depth", "num_readout_channels"]:
            probe_metadata[key] = int(probe_metadata[key])
        elif key in [
            "electrode_pitch_horz_um",
            "electrode_pitch_vert_um",
            "electrode_size_horz_direction_um",
            "shank_pitch_um",
            "shank_width_um",
            "tip_length_um",
            "even_row_horz_offset_left_edge_to_leftmost_electrode_center_um",
            "odd_row_horz_offset_left_edge_to_leftmost_electrode_center_um",
        ]:
            probe_metadata[key] = float(probe_metadata[key])

    # Read the imro table formats to find out which fields the imro tables contain
    imro_table_format_type = probe_metadata["imro_table_format_type"]
    imro_table_fields = probe_features["z_imro_formats"][imro_table_format_type + "_elm_flds"]

    # parse the imro_table_fields, which look like (value value value ...)
    list_of_imro_fields = imro_table_fields.replace("(", "").replace(")", "").split(" ")

    imro_fields_list = []
    for imro_field in list_of_imro_fields:
        imro_fields_list.append(imro_field)

    imro_fields = tuple(imro_fields_list)

    # Read MUX table information
    mux_information = None

    if "z_mux_tables" in probe_features:
        mux_table_format_type = probe_metadata.get("mux_table_format_type", None)
        mux_information = probe_features["z_mux_tables"].get(mux_table_format_type, None)

    return probe_metadata, imro_fields, mux_information


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
        # Phase3a probe does not have `ap_hp_filters` annotation
        if annotations.get("ap_hp_filters") is not None:

            for ch in range(len(data)):
                ret.append(
                    f"({ch} 0 {annotations['references'][ch]} {annotations['ap_gains'][ch]} "
                    f"{annotations['lf_gains'][ch]} {annotations['ap_hp_filters'][ch]})"
                )
        else:

            for ch in range(len(data)):
                ret.append(
                    f"({ch} 0 {annotations['references'][ch]} {annotations['ap_gains'][ch]} "
                    f"{annotations['lf_gains'][ch]})"
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


def _build_canonical_contact_id(electrode_id: int, shank_id: int | None = None) -> str:
    """
    Build the canonical contact ID string for a Neuropixels electrode.

    This establishes the standard naming convention used throughout probeinterface
    for Neuropixels contact identification.

    Parameters
    ----------
    electrode_id : int
        Physical electrode ID on the probe (e.g., 0-959 for NP1.0)
    shank_id : int or None, default: None
        Shank ID for multi-shank probes. If None, assumes single-shank probe.

    Returns
    -------
    contact_id : str
        Canonical contact ID string, either "e{electrode_id}" for single-shank
        or "s{shank_id}e{electrode_id}" for multi-shank probes.

    Examples
    --------
    >>> _build_canonical_contact_id(123)
    'e123'
    >>> _build_canonical_contact_id(123, shank_id=0)
    's0e123'
    """
    if shank_id is not None:
        return f"s{shank_id}e{electrode_id}"
    else:
        return f"e{electrode_id}"


def _parse_imro_string(imro_table_string: str, probe_part_number: str) -> dict:
    """
    Parse IMRO (Imec ReadOut) table string into structured per-channel data.

    IMRO format: "(probe_type,num_chans)(ch0 bank0 ref0 ...)(ch1 bank1 ref1 ...)..."
    Example: "(0,384)(0 1 0 500 250 1)(1 0 0 500 250 1)..."

    Note: The IMRO header contains a probe_type field (e.g., "0", "21", "24"), which is
    a numeric format version identifier that specifies which IMRO table structure was used.
    Different probe generations use different IMRO formats. This is a file format detail,
    not a physical probe property.

    Parameters
    ----------
    imro_table_string : str
        IMRO table string from SpikeGLX metadata file
    probe_part_number : str
        Probe part number (e.g., "NP1000", "NP2000")

    Returns
    -------
    imro_per_channel : dict
        Dictionary where each key maps to a list of values (one per channel).
        Keys are IMRO fields like "channel", "bank", "electrode", "ap_gain", etc.
        The "electrode" key always contains physical electrode IDs (0-959 for NP1.0, etc.).
        For NP2.0+: electrode IDs come directly from IMRO data.
        For NP1.0: electrode IDs are computed as bank * 384 + channel.
        Example: {"channel": [0,1,2,...], "bank": [1,0,0,...], "electrode": [384,1,2,...], "ap_gain": [500,500,...]}
    """
    # Get IMRO field format from catalogue
    probe_features = _load_np_probe_features()
    probe_spec = probe_features["neuropixels_probes"][probe_part_number]
    imro_format = probe_spec["imro_table_format_type"]
    imro_fields_string = probe_features["z_imro_formats"][imro_format + "_elm_flds"]
    imro_fields = tuple(imro_fields_string.replace("(", "").replace(")", "").split(" "))

    # Parse IMRO table values into per-channel data
    # Skip the header "(probe_type,num_chans)" and trailing empty string
    _, *imro_table_values_list, _ = imro_table_string.strip().split(")")
    imro_per_channel = {k: [] for k in imro_fields}
    for field_values_str in imro_table_values_list:
        values = tuple(map(int, field_values_str[1:].split(" ")))
        for field, field_value in zip(imro_fields, values):
            imro_per_channel[field].append(field_value)

    # Ensure "electrode" key always exists with physical electrode IDs
    # Different probe types encode electrode selection differently
    if "electrode" not in imro_per_channel:
        # NP1.0: Bank-based addressing (physical_electrode_id = bank * 384 + channel)
        readout_channel_ids = np.array(imro_per_channel["channel"])
        bank_key = "bank" if "bank" in imro_per_channel else "bank_mask"
        bank_indices = np.array(imro_per_channel[bank_key])
        imro_per_channel["electrode"] = (bank_indices * 384 + readout_channel_ids).tolist()

    return imro_per_channel


def read_spikeglx(file: str | Path) -> Probe:
    """
    Read probe geometry and configuration from a SpikeGLX metadata file.

    This function reconstructs the probe used in a recording by:
    1. Reading the probe part number from metadata
    2. Building the full probe geometry from manufacturer specifications
    3. Slicing to the electrodes selected in the IMRO table
    4. Further slicing to channels actually saved to disk (if subset was saved)
    5. Adding recording-specific annotations
    6. Add wiring (device channel indices)

    Parameters
    ----------
    file : Path or str
        Path to the SpikeGLX .meta file

    Returns
    -------
    probe : Probe
        Probe object with geometry, contact annotations, and device channel mapping

    See Also
    --------
    http://billkarsh.github.io/SpikeGLX/#metadata-guides

    """

    meta_file = Path(file)
    assert meta_file.suffix == ".meta", "'meta_file' should point to the .meta SpikeGLX file"

    meta = parse_spikeglx_meta(meta_file)
    assert "imroTbl" in meta, "Could not find imroTbl field in meta file!"

    # ===== 1. Extract probe part number from metadata =====
    imDatPrb_pn = meta.get("imDatPrb_pn", None)
    # Only Phase3a probe has "imProbeOpt". Map this to NP1010.
    if meta.get("imProbeOpt") is not None:
        imDatPrb_pn = "NP1010"

    # ===== 2. Build full probe with all possible contacts =====
    # This creates the complete probe geometry (e.g., 960 contacts for NP1.0)
    # based on manufacturer specifications
    full_probe = build_neuropixels_probe(probe_part_number=imDatPrb_pn)

    # ===== 3. Parse IMRO table to extract recorded electrodes and acquisition settings =====
    # IMRO = Imec ReadOut (the configuration table format from IMEC manufacturer)
    # Specifies which electrodes were selected for recording (e.g., 384 of 960) plus their
    # acquisition settings (gains, references, filters). See: https://billkarsh.github.io/SpikeGLX/help/imroTables/
    imro_table_string = meta["imroTbl"]
    imro_per_channel = _parse_imro_string(imro_table_string, imDatPrb_pn)

    # ===== 4. Build contact IDs for active electrodes =====
    # Convert physical electrode IDs to probeinterface canonical contact ID strings
    imro_electrode = imro_per_channel["electrode"]
    imro_shank = imro_per_channel.get("shank", [None] * len(imro_electrode))
    active_contact_ids = [
        _build_canonical_contact_id(elec_id, shank_id) for shank_id, elec_id in zip(imro_shank, imro_electrode)
    ]

    # ===== 5. Slice full probe to active electrodes =====
    # Find indices of active contacts in the full probe, preserving IMRO order
    contact_id_to_index = {contact_id: idx for idx, contact_id in enumerate(full_probe.contact_ids)}
    selected_contact_indices = np.array([contact_id_to_index[contact_id] for contact_id in active_contact_ids])

    probe = full_probe.get_slice(selected_contact_indices)

    # ===== 6. Store IMRO properties (acquisition settings) as annotations =====
    # Filter IMRO data to only the properties we want to add as annotations
    imro_properties_to_add = ("channel", "bank", "bank_mask", "ref_id", "ap_gain", "lf_gain", "ap_hipas_flt")
    imro_filtered = {k: v for k, v in imro_per_channel.items() if k in imro_properties_to_add and len(v) > 0}
    # Map IMRO field names to probeinterface field names and add as contact annotations
    annotations = {}
    for imro_field, values in imro_filtered.items():
        pi_field = imro_field_to_pi_field.get(imro_field)
        annotations[pi_field] = values
    probe.annotate_contacts(**annotations)

    # ===== 7. Slice to saved channels (if subset was saved) =====
    # This is DIFFERENT from IMRO selection: IMRO selects which electrodes to acquire,
    # but SpikeGLX can optionally save only a subset of acquired channels to reduce file size.
    # For example: IMRO selects 384 electrodes, but only 300 are saved to disk.
    saved_chans = get_saved_channel_indices_from_spikeglx_meta(meta_file)
    saved_chans = saved_chans[saved_chans < probe.get_contact_count()]  # Remove SYS channels
    if saved_chans.size != probe.get_contact_count():
        probe = probe.get_slice(saved_chans)

    # ===== 6. Add recording-specific annotations =====
    # These annotations identify the physical probe instance and recording setup
    imDatPrb_serial_number = meta.get("imDatPrb_sn") or meta.get("imProbeSN")  # Phase3A uses imProbeSN
    imDatPrb_port = meta.get("imDatPrb_port", None)
    imDatPrb_slot = meta.get("imDatPrb_slot", None)
    probe.annotate(serial_number=imDatPrb_serial_number)
    probe.annotate(part_number=imDatPrb_pn)
    probe.annotate(port=imDatPrb_port)
    probe.annotate(slot=imDatPrb_slot)

    # ===== 7. Set device channel indices (wiring) =====
    # I am unsure why are we are doing this. If someone knows please document it here.
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
    onix_processor = None
    for signal_chain in root.findall("SIGNALCHAIN"):
        for processor in signal_chain:
            if "PROCESSOR" == processor.tag:
                name = processor.attrib["name"]
                if "Neuropix-PXI" in name:
                    neuropix_pxi_processor = processor
                if "OneBox" in name:
                    onebox_processor = processor
                if "ONIX" in name:
                    onix_processor = processor

    if neuropix_pxi_processor is None and onebox_processor is None and onix_processor is None:
        if raise_error:
            raise Exception("Open Ephys can only be read from Neuropix-PXI, OneBox or ONIX plugins.")
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
    if onix_processor is not None:
        processor = onix_processor

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
        probe_names_used = []
        for stream_field in stream_fields:
            stream = stream_field.attrib["name"]
            # exclude ADC streams
            if "ADC" in stream:
                continue
            streams.append(stream)
            # find probe name (exclude "-AP"/"-LFP" from stream name)
            stream = stream.replace("-AP", "").replace("-LFP", "")
            if stream not in probe_names_used:
                probe_names_used.append(stream)
    else:
        has_streams = False
        probe_names_used = None

    if onix_processor is not None:
        probe_names_used = [probe_name for probe_name in probe_names_used if "Probe" in probe_name]

    # for Open Ephys version < 1.0 np_probes is in the EDITOR field.
    # for Open Ephys version >= 1.0 np_probes is in the CUSTOM_PARAMETERS field.
    editor = processor.find("EDITOR")
    if oe_version < parse("0.9.0"):
        np_probes = editor.findall("NP_PROBE")
    else:
        custom_parameters = editor.find("CUSTOM_PARAMETERS")
        if onix_processor is not None:
            possible_probe_names = ["NEUROPIXELSV1E", "NEUROPIXELSV1F", "NEUROPIXELSV2E"]
            parent_np_probe = ""
            for possible_probe_name in possible_probe_names:
                parent_np_probe = custom_parameters.findall(possible_probe_name)
                if len(parent_np_probe) > 0:
                    break
            if possible_probe_name == "NEUROPIXELSV2E":
                np_probes = [parent_np_probe[0].findall(f"PROBE{a}")[0] for a in range(2)]
            else:
                np_probes = [parent_np_probe[0]]
        else:
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
                raise Exception(f"Not enough NP_PROBE entries ({len(np_probes)}) for used probes: {probe_names_used}")
            return None

    probe_features = _load_np_probe_features()

    np_probes_info = []

    # now load probe info from NP_PROBE fields
    np_probes_info = []
    for probe_idx, np_probe in enumerate(np_probes):
        # selected_electrodes is the preferred way to instantiate the probe
        # if this field is available, a full probe is created from the probe_part_number
        # and then sliced using the selected electrodes.
        # if not available, the xpos and ypos fields are used to create the probe
        slot = np_probe.attrib.get("slot")
        port = np_probe.attrib.get("port")
        dock = np_probe.attrib.get("dock")
        probe_part_number = np_probe.attrib.get("probe_part_number") or np_probe.attrib.get("probePartNumber")
        probe_serial_number = np_probe.attrib.get("probe_serial_number") or np_probe.attrib.get("probeSerialNumber")
        selected_electrodes = np_probe.find("SELECTED_ELECTRODES")
        channels = np_probe.find("CHANNELS")

        pt_metadata, _, mux_info = get_probe_metadata_from_probe_features(probe_features, probe_part_number)

        if selected_electrodes is not None:
            selected_electrodes_values = selected_electrodes.attrib.values()

            num_shank = pt_metadata["num_shanks"]
            contact_per_shank = pt_metadata["cols_per_shank"] * pt_metadata["rows_per_shank"]

            if num_shank == 1:
                elec_ids = np.arange(contact_per_shank, dtype=int)
                shank_ids = None
            else:
                elec_ids = np.concatenate([np.arange(contact_per_shank, dtype=int) for i in range(num_shank)])
                shank_ids = np.concatenate([np.zeros(contact_per_shank, dtype=int) + i for i in range(num_shank)])

            full_probe = _make_npx_probe_from_description(
                pt_metadata, probe_part_number, elec_ids, shank_ids, mux_info=mux_info
            )

            selected_electrode_indices = [int(electrode_index) for electrode_index in selected_electrodes_values]

            sliced_probe = full_probe.get_slice(selection=selected_electrode_indices)

            np_probe_dict = {
                "pt_metadata": pt_metadata,
                "serial_number": probe_serial_number,
                "part_number": probe_part_number,
                "mux_info": mux_info,
                "probe": sliced_probe,
            }
        else:

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

            shank_pitch = pt_metadata["shank_pitch_um"]

            if fix_x_position_for_oe_5 and oe_version < parse("0.6.0") and shank_ids is not None:
                positions[:, 1] = positions[:, 1] - shank_pitch * shank_ids

            # x offset so that the first column is at 0x
            offset = np.min(positions[:, 0])
            # if some shanks are not used, we need to adjust the offset
            if shank_ids is not None:
                offset -= np.min(shank_ids) * shank_pitch
            positions[:, 0] -= offset

            #
            y_pitch = pt_metadata[
                "electrode_pitch_vert_um"
            ]  # Vertical spacing between the centers of adjacent contacts
            x_pitch = pt_metadata[
                "electrode_pitch_horz_um"
            ]  # Horizontal spacing between the centers of contacts within the same row
            number_of_columns = pt_metadata["cols_per_shank"]
            probe_stagger = (
                pt_metadata["even_row_horz_offset_left_edge_to_leftmost_electrode_center_um"]
                - pt_metadata["odd_row_horz_offset_left_edge_to_leftmost_electrode_center_um"]
            )
            num_shanks = pt_metadata["num_shanks"]

            description = pt_metadata.get("description")

            elec_ids = []
            for i, pos in enumerate(positions):
                # Do not calculate contact ids if the model name is not known
                if description is None:
                    elec_ids = None
                    break

                x_pos = pos[0]
                y_pos = pos[1]

                # Adds a shift to rows in the staggered configuration
                is_row_staggered = np.mod(y_pos / y_pitch + 1, 2) == 1
                row_stagger = probe_stagger if is_row_staggered else 0

                # Map the positions to the contacts ids
                shank_id = shank_ids[i] if num_shanks > 1 else 0

                # Electrode ids are computed from the positions of the electrodes. The computation
                # is different for probes with one row of electrodes, or more than one.
                if x_pitch == 0:
                    elec_id = int(number_of_columns * y_pos / y_pitch)
                else:
                    elec_id = int(
                        (x_pos - row_stagger - shank_pitch * shank_id) / x_pitch + number_of_columns * y_pos / y_pitch
                    )
                elec_ids.append(elec_id)

            np_probe_dict = {
                "shank_ids": shank_ids,
                "elec_ids": elec_ids,
                "pt_metadata": pt_metadata,
                "slot": slot,
                "port": port,
                "dock": dock,
                "serial_number": probe_serial_number,
                "part_number": probe_part_number,
                "mux_info": mux_info,
            }

        # Sequentially assign probe names
        if "custom_probe_name" in np_probe.attrib and np_probe.attrib["custom_probe_name"] != probe_serial_number:
            name = np_probe.attrib["custom_probe_name"]
        else:
            name = probe_names_used[probe_idx]
        np_probe_dict.update({"name": name})
        np_probes_info.append(np_probe_dict)

    # now select find the selected probe (if multiple)
    if len(np_probes) > 1:
        found = False
        probe_names = [p["name"] for p in np_probes_info]

        if stream_name is not None:
            assert probe_name is None and serial_number is None, (
                "Use one of 'stream_name', 'probe_name', " "or 'serial_number'"
            )
            # Here we have to check if the probe name or the serial number is in the stream name
            # If both are present, e.g., the name contains the serial number, the first match is used.
            for probe_idx, probe_info in enumerate(np_probes_info):
                if probe_info["name"] in stream_name:
                    found = True
                    break
            if not found:
                for probe_idx, probe_info in enumerate(np_probes_info):
                    if probe_info["serial_number"] in stream_name:
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
    probe = np_probe_info.get("probe")

    if probe is None:
        # check if subset of channels
        shank_ids = np_probe_info["shank_ids"]
        elec_ids = np_probe_info["elec_ids"]
        pt_metadata = np_probe_info["pt_metadata"]
        mux_info = np_probe_info["mux_info"]

        probe = _make_npx_probe_from_description(
            pt_metadata, probe_part_number, elec_ids, shank_ids=shank_ids, mux_info=mux_info
        )

    chans_saved = get_saved_channel_indices_from_openephys_settings(settings_file, stream_name=stream_name)
    if chans_saved is not None:
        probe = probe.get_slice(chans_saved)

    probe.serial_number = np_probe_info["serial_number"]
    probe.name = np_probe_info["name"]

    probe.annotate(
        part_number=np_probe_info["part_number"],
    )
    if "slot" in np_probe_info:
        probe.annotate(slot=np_probe_info["slot"])
    if "port" in np_probe_info:
        probe.annotate(port=np_probe_info["port"])
    if "dock" in np_probe_info:
        probe.annotate(dock=np_probe_info["dock"])

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
                                f"found. Using first one"
                            )
                        custom_stream = possible_custom_streams[0]
                    else:
                        custom_stream = custom_streams[0]
                    recording_state = custom_stream.attrib.get("recording_state", None)
                    if recording_state is not None:
                        if recording_state not in ("ALL", "NONE"):
                            chans_saved = np.array([chan for chan, r in enumerate(recording_state) if int(r) == 1])
    return chans_saved
