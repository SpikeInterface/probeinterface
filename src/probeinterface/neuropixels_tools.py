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
        with open(probe_features_filepath, "r") as f:
            _np_probe_features = json.load(f)
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
    Returns the length of a given probe from ProbeTable specifications.

    Parameters
    ----------
    probe_part_number : str
        The part number of the probe e.g. 'NP2013'.

    Returns
    -------
    probe_length : int
        Length of full probe shank from tip to base (microns)
    """
    np_features = _load_np_probe_features()
    probe_spec = np_features["neuropixels_probes"].get(probe_part_number)

    if probe_spec is not None and "shank_tip_to_base_um" in probe_spec:
        return int(probe_spec["shank_tip_to_base_um"])

    # Fallback for unknown probes or missing field
    return 10_000


def make_mux_table_array(mux_information) -> np.array:
    """
    Function to parse the mux_table from ProbeTable.

    Parameters
    ----------
    mux_information : str
        The information from `z_mux_tables` in the ProbeTable `probe_feature.json` file

    Returns
    -------
    num_adcs: int
        Number of ADCs used in the probe's readout system.
    num_channels_per_adc: int
        Number of readout channels assigned to each ADC.
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
    Read a Neuropixels probe from an IMRO (Imec ReadOut) table file.

    IMRO files (.imro) are used by SpikeGLX and Open Ephys to configure which electrodes
    are recorded and their acquisition settings (gains, references, filters). This function
    reads the file, determines the probe part number from the IMRO header, builds the full
    catalogue probe, and slices it to the active electrodes specified in the table.

    Parameters
    ----------
    file_path : Path or str
        The .imro file path

    Returns
    -------
    probe : Probe
        Probe object with geometry, contact annotations, and device channel mapping.

    See Also
    --------
    https://billkarsh.github.io/SpikeGLX/help/imroTables/

    """
    # ===== 1. Read file and determine probe part number from IMRO header =====
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

    # ===== 2. Interpret IMRO table =====
    imro_per_channel = _parse_imro_string(imro_str, imDatPrb_pn)

    # ===== 3. Build full probe with all possible contacts =====
    full_probe = build_neuropixels_probe(probe_part_number=imDatPrb_pn)

    # ===== 4. Build contact IDs for active electrodes =====
    active_contact_ids = _get_active_contact_ids(imro_per_channel)

    # ===== 5. Slice full probe to active electrodes =====
    contact_id_to_index = {cid: i for i, cid in enumerate(full_probe.contact_ids)}
    selected_indices = np.array([contact_id_to_index[cid] for cid in active_contact_ids])
    probe = full_probe.get_slice(selected_indices)

    # ===== 7. Annotate probe with recording-specific metadata =====
    adc_sampling_table = probe.annotations.get("adc_sampling_table")
    _annotate_probe_with_adc_sampling_info(probe, adc_sampling_table)

    # Scalar annotations
    probe_type = imro_str.strip().split(")")[0].split(",")[0][1:]
    probe.annotate(probe_type=probe_type)

    # Vector annotations from IMRO fields
    vector_properties = ("channel", "bank", "bank_mask", "ref_id", "ap_gain", "lf_gain", "ap_hipas_flt")
    vector_properties_available = {}
    for k, v in imro_per_channel.items():
        if k in vector_properties and len(v) > 0:
            vector_properties_available[imro_field_to_pi_field.get(k)] = v
    probe.annotate_contacts(**vector_properties_available)

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

    # ===== 8. Store ADC sampling table =====
    # The ADC sampling table describes how readout channels map to ADCs, not electrodes.
    # Per-contact annotations (adc_group, adc_sample_order) can only be correctly
    # assigned when reading a recording with a known channel map (via read_spikeglx),
    # because the table indices are readout channel indices, not electrode indices.
    # We store the full table string so it's available after slicing.
    mux_table_format_type = probe_spec_dict["mux_table_format_type"]
    adc_sampling_table = probe_features["z_mux_tables"].get(mux_table_format_type)
    if adc_sampling_table is not None:
        probe.annotate(adc_sampling_table=adc_sampling_table)

    return probe


def _annotate_contacts_from_mux_table(probe: Probe, adc_groups_array: np.array):
    """
    Annotate a Probe object with ADC group and sample order information based on the MUX table.

    Neuropixels probes multiplex their electrodes through a fixed set of ADCs. For
    example, an NP2000 probe has 24 ADCs, each sampling 16 readout channels in
    sequence. The ``adc_groups_array`` encodes this mapping: each row is one sampling
    time slot, and the values are readout channel numbers that are sampled
    simultaneously (one per ADC). For example, if row 0 contains
    ``[0, 1, 32, 33, 64, 65, ...]``, it means readout channels 0, 1, 32, 33, 64, 65
    are all sampled at the same time by different ADCs.

    This function uses those readout channel numbers directly as indices into the
    probe's contact array to assign two per-contact annotations:

    - ``adc_group``: which ADC samples this contact (column index in the table)
    - ``adc_sample_order``: at which time step within the ADC's cycle this contact
      is sampled (row index in the table)

    This means contact index ``i`` in the probe must correspond to readout channel
    ``i``. This holds when the probe has been sliced in readout channel order, which
    is the case for both ``read_spikeglx`` (IMRO table lists channels in readout
    order 0-383) and ``read_openephys`` (channels sorted by ``CH`` number, which
    corresponds to readout channel number). If the probe contacts were reordered
    (e.g., sorted by position on the shank), these annotations would be wrong.

    Parameters
    ----------
    probe : Probe
        The Probe object to annotate. Contacts must be in readout channel order.
    adc_groups_array : np.array
        Array shaped ``(num_time_slots, num_adcs)`` where each value is a readout
        channel number. Readout channel at ``adc_groups_array[slot, adc]`` is
        sampled by ADC ``adc`` at time slot ``slot``.
    """
    # Map readout channels to ADC groups and sample order.
    # The indices in adc_groups_array are readout channel numbers, and we
    # use them directly as contact indices into the probe.
    num_readout_channels = probe.get_contact_count()
    adc_groups = np.zeros(num_readout_channels, dtype="int64")
    adc_sample_order = np.zeros(num_readout_channels, dtype="int64")
    for adc_index, channels_per_adc in enumerate(adc_groups_array):
        # Filter out placeholder values (e.g., 128 in mux_np1200 for unused slots)
        valid_channels = channels_per_adc[channels_per_adc < num_readout_channels]
        adc_groups[valid_channels] = adc_index
        adc_sample_order[valid_channels] = np.arange(len(valid_channels))

    probe.annotate_contacts(adc_group=adc_groups)
    probe.annotate_contacts(adc_sample_order=adc_sample_order)


def _annotate_probe_with_adc_sampling_info(probe: Probe, adc_sampling_table: str | None):
    """
    Annotate a Probe object with ADC group and sample order information based on the ADC sampling table.

    This function is used when reading a recording with a known channel map (via read_spikeglx, read_openephys)
    to assign per-contact annotations for adc_group and adc_sample_order, which describe how
    each contact maps to the ADCs during recording, and global annotations for num_adcs and num_channels_per_adc.

    Parameters
    ----------
    probe : Probe
        The Probe object to annotate. Must have device_channel_indices set.
    adc_sampling_table : str
        The ADC sampling table string from the probe features, which describes how readout channels map to ADCs.

    Returns
    -------
    None
        The function modifies the Probe object in place.
    """
    # Parse table string: (num_adcs,num_channels_per_adc)(ch ch ...)(ch ch ...)...
    if adc_sampling_table is None:
        return
    adc_info = adc_sampling_table.split(")(")[0]
    split_mux = adc_sampling_table.split(")(")[1:]
    num_adcs, num_channels_per_adc = map(int, adc_info[1:].split(","))
    probe.annotate(num_adcs=num_adcs)
    probe.annotate(num_channels_per_adc=num_channels_per_adc)

    adc_groups_list = [
        np.array(each_mux.replace("(", "").replace(")", "").split(" ")).astype("int") for each_mux in split_mux
    ]
    adc_groups_array = np.transpose(np.array(adc_groups_list))

    # Map readout channels to ADC groups and sample order
    _annotate_contacts_from_mux_table(probe, adc_groups_array)


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
        Dictionary with a "header" key containing parsed header fields, and one key per
        IMRO entry field mapping to a list of values. The number of entries varies by probe
        type (384 per channel for most probes, 24 per group for NP1110).
        NP2.x+ probes will have an "electrode" key directly. NP1.x probes will not
        (electrode IDs must be resolved separately via _get_active_contact_ids).
        Example for NP1.0: {"header": {"type": 0, "num_channels": 384},
            "channel": [0,1,...], "bank": [0,0,...], "ref_id": [0,0,...], ...}
        Example for NP2.0: {"header": {"type": 21, "num_channels": 384},
            "channel": [0,1,...], "bank_mask": [1,1,...], "electrode": [0,1,...], ...}
        Example for NP1110: {"header": {"type": 1110, "col_mode": 2, "ref_id": 0, ...},
            "group": [0,1,...], "bankA": [0,0,...], "bankB": [0,0,...]}  # 24 entries, not 384
    """
    # Get IMRO field format from catalogue
    probe_features = _load_np_probe_features()
    probe_spec = probe_features["neuropixels_probes"][probe_part_number]
    imro_format = probe_spec["imro_table_format_type"]
    imro_fields_string = probe_features["z_imro_formats"][imro_format + "_elm_flds"]
    imro_fields = tuple(imro_fields_string.replace("(", "").replace(")", "").split(" "))

    # Parse IMRO header and per-entry values
    header_str, *imro_table_values_list, _ = imro_table_string.strip().split(")")

    # Parse header fields using the catalogue schema
    imro_header_fields_string = probe_features["z_imro_formats"][imro_format + "_hdr_flds"]
    imro_header_fields = tuple(imro_header_fields_string.replace("(", "").replace(")", "").split(","))
    header_values = tuple(map(int, header_str[1:].split(",")))
    # Initialize with parsed header and empty lists for per-entry fields (filled below)
    imro_per_channel = {"header": dict(zip(imro_header_fields, header_values))}
    for field in imro_fields:
        imro_per_channel[field] = []
    for field_values_str in imro_table_values_list:
        values = tuple(map(int, field_values_str[1:].split(" ")))
        for field, field_value in zip(imro_fields, values):
            imro_per_channel[field].append(field_value)

    return imro_per_channel


def _get_active_contact_ids(imro_per_channel: dict) -> list[str]:
    """
    Get canonical contact ID strings for the active electrodes in a parsed IMRO table.

    If the IMRO format includes electrode IDs directly (NP2.x+), uses them as-is.
    If not (NP1.x), resolves them first via the appropriate addressing scheme
    (simple bank for NP1.0-like probes, UHD group-based for NP1110).

    Parameters
    ----------
    imro_per_channel : dict
        Parsed IMRO data from _parse_imro_string. Modified in place if electrode
        IDs need to be resolved.

    Returns
    -------
    list of str
        Canonical contact ID strings (e.g., ["e0", "e384", ...] or ["s0e123", ...]).
    """
    if "electrode" not in imro_per_channel:
        _resolve_active_contacts_for_np1(imro_per_channel)
    if "electrode" not in imro_per_channel:
        _resolve_active_contacts_for_np1110(imro_per_channel)
    assert "electrode" in imro_per_channel, (
        f"Could not resolve electrode IDs from IMRO fields: {list(imro_per_channel.keys())}"
    )

    elec_ids = imro_per_channel["electrode"]
    shank_ids = imro_per_channel.get("shank", [None] * len(elec_ids))
    return [
        _build_canonical_contact_id(elec_id, shank_id)
        for shank_id, elec_id in zip(shank_ids, elec_ids)
    ]


def _resolve_active_contacts_for_np1(imro_per_channel: dict) -> None:
    """
    Compute electrode IDs for NP 1.0-like probes that use simple bank addressing.

    These probes (IMRO types 0, 1020, 1030, 1100, 1120-1123, 1200, 1300) have
    "channel" and "bank" fields in their IMRO table but no "electrode" field.
    The electrode ID is computed as: electrode = bank * 384 + channel.

    Modifies imro_per_channel in place, adding the "electrode" key.

    Parameters
    ----------
    imro_per_channel : dict
        Parsed IMRO data from _parse_imro_string. Modified in place.
    """
    if "channel" not in imro_per_channel:
        return

    readout_channel_ids = np.array(imro_per_channel["channel"])
    bank_key = "bank" if "bank" in imro_per_channel else "bank_mask"
    bank_indices = np.array(imro_per_channel[bank_key])
    imro_per_channel["electrode"] = (bank_indices * 384 + readout_channel_ids).tolist()


def _resolve_active_contacts_for_np1110(imro_per_channel: dict) -> None:
    """
    Compute electrode IDs for NP1110 (UHD2 active) probes that use group-based addressing.

    NP1110 has 6144 electrodes in an 8x768 grid, 384 readout channels, 24 groups, and 16 banks.
    The IMRO table has 24 per-group entries (group, bankA, bankB) and a header with col_mode.
    Each group covers 16 channels. For each channel, the group index is deterministic, the
    effective bank is selected from (bankA, bankB) based on col_mode, and the electrode ID
    is computed from channel + bank using column/row lookup tables.

    Modifies imro_per_channel in place, adding the "electrode" key with 384 electrode IDs.

    Parameters
    ----------
    imro_per_channel : dict
        Parsed IMRO data with keys "group", "bankA", "bankB" (24 entries each)
        and "header" dict containing "col_mode".

    References
    ----------
    https://github.com/billkarsh/SpikeGLX/blob/51b96c70204c025748d69c9a588e07406728f9eb/Src-imro/IMROTbl_T1110.cpp
    """
    if "group" not in imro_per_channel:
        return

    # TODO: Remove this warning once we have test data for NP1110 recordings.
    warnings.warn(
        "NP1110 (Neuropixels 1.0 UHD2 active) support is experimental. "
        "The active electrode selection logic is translated directly from SpikeGLX "
        "(https://github.com/billkarsh/SpikeGLX, Src-imro/IMROTbl_T1110.cpp) but has not "
        "been validated against real NP1110 recordings. Please double-check the electrode "
        "selection and report any issues at https://github.com/SpikeInterface/probeinterface/issues",
        UserWarning,
        stacklevel=3,  # Points to read_imro / read_spikeglx (caller of _ensure_active_contacts_available)
    )

    col_mode = imro_per_channel["header"]["col_mode"]  # 0=INNER, 1=OUTER, 2=ALL

    groups_bankA = imro_per_channel["bankA"]
    groups_bankB = imro_per_channel["bankB"]

    # With pain in my heart, I am following here the C++ convention of terse naming from the
    # original SpikeGLX implementation (IMROTbl_T1110.cpp). The purpose is to make it easy to
    # spot differences when comparing against the original code, until we have real NP1110 test
    # data to validate against and feel comfortable (if ever) renaming to our own conventions.
    col_tbl = [0, 3, 1, 2, 1, 2, 0, 3]

    def grpIdx(ch):
        return 2 * ((ch % 384) // 32) + ((ch % 384) & 1)

    def col(ch, bank):
        grp_col = col_tbl[4 * (bank & 1) + (grpIdx(ch) % 4)]
        crossed = (bank // 4) & 1
        ingrp_col = ((((ch % 64) % 32) // 2) & 1) ^ crossed
        if ch & 1:
            return 2 * grp_col + (1 - ingrp_col)
        else:
            return 2 * grp_col + ingrp_col

    def row(ch, bank):
        grp_row = grpIdx(ch) // 4
        ingrp_row = ((ch % 64) % 32) // 4
        if ch & 1:
            b0_row = 8 * grp_row + (7 - ingrp_row)
        else:
            b0_row = 8 * grp_row + ingrp_row
        return 48 * bank + b0_row

    def bank(ch, bankA, bankB):
        if col_mode == 2:  # ALL
            return bankA
        # INNER (0) or OUTER (1): choose bankA or bankB based on column position
        c = col(ch, bankA)
        if c <= 3:
            if col_mode == 1:  # OUTER
                return bankA if not (c & 1) else bankB
            else:  # INNER
                return bankA if (c & 1) else bankB
        else:
            if col_mode == 1:  # OUTER
                return bankA if (c & 1) else bankB
            else:  # INNER
                return bankA if not (c & 1) else bankB

    electrode_ids = []
    for ch in range(384):
        grp = grpIdx(ch)
        b = bank(ch, groups_bankA[grp], groups_bankB[grp])
        electrode_ids.append(8 * row(ch, b) + col(ch, b))

    imro_per_channel["electrode"] = electrode_ids
    # Also add the "channel" key (0-383) since the IMRO entries are per-group, not per-channel
    imro_per_channel["channel"] = list(range(384))


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
    active_contact_ids = _get_active_contact_ids(imro_per_channel)

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

    # ===== 6b. Add ADC sampling annotations =====
    # The ADC sampling table describes which ADC samples each readout channel and in what order.
    # At this point, contacts are ordered by readout channel (0-383), so we can directly
    # apply the mapping. This must be done here (not in build_neuropixels_probe)
    # because the table indices are readout channel indices, not electrode indices.
    adc_sampling_table = probe.annotations.get("adc_sampling_table")
    _annotate_probe_with_adc_sampling_info(probe, adc_sampling_table)

    # ===== 7. Slice to saved channels (if subset was saved) =====
    # This is DIFFERENT from IMRO selection: IMRO selects which electrodes to acquire,
    # but SpikeGLX can optionally save only a subset of acquired channels to reduce file size.
    # For example: IMRO selects 384 electrodes, but only 300 are saved to disk.
    saved_chans = get_saved_channel_indices_from_spikeglx_meta(meta_file)
    saved_chans = saved_chans[saved_chans < probe.get_contact_count()]  # Remove SYS channels
    if saved_chans.size != probe.get_contact_count():
        probe = probe.get_slice(saved_chans)

    # ===== 8. Add recording-specific annotations =====
    # These annotations identify the physical probe instance and recording setup
    imDatPrb_serial_number = meta.get("imDatPrb_sn") or meta.get("imProbeSN")  # Phase3A uses imProbeSN
    imDatPrb_port = meta.get("imDatPrb_port", None)
    imDatPrb_slot = meta.get("imDatPrb_slot", None)
    probe.annotate(serial_number=imDatPrb_serial_number)
    probe.annotate(part_number=imDatPrb_pn)
    probe.annotate(port=imDatPrb_port)
    probe.annotate(slot=imDatPrb_slot)

    # ===== 9. Set device channel indices (wiring) =====
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


def _parse_openephys_settings(
    settings_file: str | Path,
    fix_x_position_for_oe_5: bool = True,
    raise_error: bool = True,
) -> Optional[list[dict]]:
    """
    Parse an Open Ephys settings.xml and extract per-probe metadata.

    Returns a list of dicts, one per enabled probe, containing the probe_part_number,
    serial_number, name, slot/port/dock, and electrode selection info needed to
    build the probe from the catalogue.

    Parameters
    ----------
    settings_file : Path or str
        Path to the Open Ephys settings.xml file.
    fix_x_position_for_oe_5 : bool
        Fix position bug in open-ephys < 0.6.0.
    raise_error : bool
        If True, raise on error. If False, return None.

    Returns
    -------
    probes_info : list of dict, or None
        Each dict contains:
        - probe_part_number, serial_number, name, slot, port, dock
        - selected_electrode_indices: list of int (from SELECTED_ELECTRODES), or None
        - contact_ids: list of str (reverse-engineered from CHANNELS), or None
        - settings_channel_keys: np.array of str, or None
        - elec_ids, shank_ids, pt_metadata, mux_info: for legacy fallback
    """
    ET = import_safely("xml.etree.ElementTree")
    tree = ET.parse(str(settings_file))
    root = tree.getroot()

    info_chain = root.find("INFO")
    oe_version = parse(info_chain.find("VERSION").text)
    neuropix_pxi_processor = None
    onebox_processor = None
    onix_processor = None
    channel_map = None
    record_node = None
    channel_map_position = None
    record_node_position = None
    proc_counter = 0

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
                if "Channel Map" in name:
                    channel_map = processor
                    channel_map_position = proc_counter
                if "Record Node" in name:
                    record_node = processor
                    record_node_position = proc_counter
                proc_counter += 1

    # Check if Channel Map comes before Record Node
    channel_map_before_record_node = (
        channel_map_position is not None
        and record_node_position is not None
        and channel_map_position < record_node_position
    )

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

    # Read STREAM fields if present (>=0.6.x)
    stream_fields = processor.findall("STREAM")
    if len(stream_fields) > 0:
        has_streams = True
        probe_names_used = []
        for stream_field in stream_fields:
            stream = stream_field.attrib["name"]
            # exclude ADC streams
            if "ADC" in stream:
                continue
            # find probe name (exclude "-AP"/"-LFP" from stream name)
            stream = stream.replace("-AP", "").replace("-LFP", "")
            if stream not in probe_names_used:
                probe_names_used.append(stream)
    else:
        has_streams = False
        probe_names_used = None

    if onix_processor is not None:
        probe_names_used = [pn for pn in probe_names_used if "Probe" in pn]

    # Load custom channel maps, if channel map is present and comes before record node
    # (if not, it won't be applied to the recording)
    probe_custom_channel_maps = None
    if channel_map is not None and channel_map_before_record_node:
        stream_fields = channel_map.findall("STREAM")
        custom_parameters = channel_map.findall("CUSTOM_PARAMETERS")
        if custom_parameters is not None:
            custom_parameters = custom_parameters[0]
            custom_maps_all = custom_parameters.findall("STREAM")
            probe_custom_channel_maps = []
            # filter ADC streams and keep custom maps for probe streams
            for i, stream_field in enumerate(stream_fields):
                stream = stream_field.attrib["name"]
                # exclude ADC streams
                if "ADC" in stream:
                    continue
                custom_indices = [int(ch.attrib["index"]) for ch in custom_maps_all[i].findall("CH")]
                probe_custom_channel_maps.append(custom_indices)

    # For Open Ephys version < 1.0 np_probes is in the EDITOR field.
    # For Open Ephys version >= 1.0 np_probes is in the CUSTOM_PARAMETERS field.
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

    # If STREAMs are not available, probes are sequentially named based on the node id
    if not has_streams:
        probe_names_used = [f"{node_id}.{stream_index}" for stream_index in range(len(np_probes))]

    # Check consistency with stream names and other fields
    if has_streams:
        if len(np_probes) < len(probe_names_used):
            if raise_error:
                raise Exception(f"Not enough NP_PROBE entries ({len(np_probes)}) for used probes: {probe_names_used}")
            return None

    probe_features = _load_np_probe_features()

    probes_info = []
    for probe_idx, np_probe in enumerate(np_probes):
        slot = np_probe.attrib.get("slot")
        port = np_probe.attrib.get("port")
        dock = np_probe.attrib.get("dock")
        probe_part_number = np_probe.attrib.get("probe_part_number") or np_probe.attrib.get("probePartNumber")
        probe_serial_number = np_probe.attrib.get("probe_serial_number") or np_probe.attrib.get("probeSerialNumber")
        selected_electrodes = np_probe.find("SELECTED_ELECTRODES")
        channels = np_probe.find("CHANNELS")

        pt_metadata, _, mux_info = get_probe_metadata_from_probe_features(probe_features, probe_part_number)

        # Assign probe name
        if "custom_probe_name" in np_probe.attrib and np_probe.attrib["custom_probe_name"] != probe_serial_number:
            name = np_probe.attrib["custom_probe_name"]
        else:
            name = probe_names_used[probe_idx]

        info = {
            "probe_part_number": probe_part_number,
            "serial_number": probe_serial_number,
            "name": name,
            "slot": slot,
            "port": port,
            "dock": dock,
            "pt_metadata": pt_metadata,
            "mux_info": mux_info,
            "selected_electrode_indices": None,
            "contact_ids": None,
            "settings_channel_keys": None,
            "elec_ids": None,
            "shank_ids": None,
            "custom_channel_map": None,
        }

        if selected_electrodes is not None:
            # Newer plugin versions provide electrode indices directly
            info["selected_electrode_indices"] = [int(ei) for ei in selected_electrodes.attrib.values()]
            if probe_custom_channel_maps is not None:
                # Slice custom channel maps to match the number of selected electrodes
                # (required when SYNC channel is present)
                custom_indices = probe_custom_channel_maps[probe_idx][: len(info["selected_electrode_indices"])]
                info["custom_channel_map"] = custom_indices
        else:
            # Older plugin versions: reverse-engineer electrode IDs from positions
            channel_names = np.array(list(channels.attrib.keys()))
            channel_ids = np.array([int(ch[2:]) for ch in channel_names])
            channel_order = np.argsort(channel_ids)

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
            if shank_ids is not None:
                offset -= np.min(shank_ids) * shank_pitch
            positions[:, 0] -= offset

            y_pitch = pt_metadata["electrode_pitch_vert_um"]
            x_pitch = pt_metadata["electrode_pitch_horz_um"]
            number_of_columns = pt_metadata["cols_per_shank"]
            probe_stagger = (
                pt_metadata["even_row_horz_offset_left_edge_to_leftmost_electrode_center_um"]
                - pt_metadata["odd_row_horz_offset_left_edge_to_leftmost_electrode_center_um"]
            )
            num_shanks = pt_metadata["num_shanks"]
            description = pt_metadata.get("description")

            elec_ids = []
            for i, pos in enumerate(positions):
                if description is None:
                    elec_ids = None
                    break

                x_pos = pos[0]
                y_pos = pos[1]

                is_row_staggered = np.mod(y_pos / y_pitch + 1, 2) == 1
                row_stagger = probe_stagger if is_row_staggered else 0

                shank_id = shank_ids[i] if num_shanks > 1 else 0

                if x_pitch == 0:
                    elec_id = int(number_of_columns * y_pos / y_pitch)
                else:
                    elec_id = int(
                        (x_pos - row_stagger - shank_pitch * shank_id) / x_pitch + number_of_columns * y_pos / y_pitch
                    )
                elec_ids.append(elec_id)

            info["settings_channel_keys"] = channel_names
            info["shank_ids"] = shank_ids
            info["elec_ids"] = elec_ids

            # Build contact_ids from reverse-engineered electrode IDs
            if elec_ids is not None:
                shank_ids_iter = shank_ids if shank_ids is not None else [None] * len(elec_ids)
                info["contact_ids"] = [
                    _build_canonical_contact_id(eid, sid) for sid, eid in zip(shank_ids_iter, elec_ids)
                ]
            if probe_custom_channel_maps is not None:
                # Slice custom channel maps to match the number of selected electrodes
                # (required when SYNC channel is present)
                custom_indices = probe_custom_channel_maps[probe_idx][: len(info["contact_ids"])]
                info["custom_channel_map"] = custom_indices

        probes_info.append(info)

    return probes_info


def _select_openephys_probe_info(
    probes_info: list[dict],
    stream_name: Optional[str] = None,
    probe_name: Optional[str] = None,
    serial_number: Optional[str] = None,
    raise_error: bool = True,
) -> Optional[dict]:
    """
    Select one probe's info dict from the list returned by `_parse_openephys_settings`.

    Parameters
    ----------
    probes_info : list of dict
        List of per-probe info dicts from `_parse_openephys_settings`.
    stream_name : str or None
        Stream name for selection.
    probe_name : str or None
        Probe name for selection.
    serial_number : str or None
        Serial number for selection.
    raise_error : bool
        If True, raise on error. If False, return None.

    Returns
    -------
    info : dict or None
    """
    probe_names_used = [p["name"] for p in probes_info]

    if len(probes_info) > 1:
        found = False

        if stream_name is not None:
            assert probe_name is None and serial_number is None, (
                "Use one of 'stream_name', 'probe_name', " "or 'serial_number'"
            )
            for probe_info in probes_info:
                if probe_info["name"] in stream_name:
                    found = True
                    break
            if not found:
                for probe_info in probes_info:
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
            for probe_info in probes_info:
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
            for probe_info in probes_info:
                if probe_info["serial_number"] == str(serial_number):
                    found = True
                    break
            if not found:
                np_serial_numbers = [p["serial_number"] for p in probes_info]
                if raise_error:
                    raise Exception(
                        f"The provided {serial_number} is not in the available serial numbers: {np_serial_numbers}"
                    )
                return None
        else:
            raise Exception(
                f"More than one probe found. Use one of 'stream_name', 'probe_name', or 'serial_number' "
                f"to select the right probe.\nProbe names: {probe_names_used}"
            )
        return probe_info
    else:
        available_probe_name = probes_info[0]["name"]
        available_serial_number = probes_info[0]["serial_number"]

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
        return probes_info[0]


def _slice_openephys_catalogue_probe(full_probe: Probe, probe_info: dict) -> Probe:
    """
    Slice a full catalogue probe using the electrode selection from probe_info.

    For SELECTED_ELECTRODES (newer plugin), uses the indices directly.
    For CHANNELS (older plugin), matches reverse-engineered contact_ids to the catalogue.

    If the `custom_channel_map` field is present in probe_info, due to a "Channel Map" processor in the signal
    chain that comes before the "Record Node", it is applied as a further slice after electrode selection.

    Parameters
    ----------
    full_probe : Probe
        Full catalogue probe from `build_neuropixels_probe`.
    probe_info : dict
        Probe info dict from `_parse_openephys_settings`.

    Returns
    -------
    probe : Probe
    """
    custom_channel_map = probe_info.get("custom_channel_map")
    if probe_info["selected_electrode_indices"] is not None:
        selected_electrode_indices = np.array(probe_info["selected_electrode_indices"])
        if custom_channel_map is not None:
            selected_electrode_indices = selected_electrode_indices[custom_channel_map]
        return full_probe.get_slice(selection=selected_electrode_indices)

    contact_ids = probe_info["contact_ids"]
    if contact_ids is not None:
        catalogue_ids = set(full_probe.contact_ids)
        if all(cid in catalogue_ids for cid in contact_ids):
            contact_id_to_index = {cid: i for i, cid in enumerate(full_probe.contact_ids)}
            selected_indices = np.array([contact_id_to_index[cid] for cid in contact_ids])
            if custom_channel_map is not None:
                selected_indices = selected_indices[custom_channel_map]
            return full_probe.get_slice(selection=selected_indices)
        else:
            raise ValueError(
                f"Could not match electrode positions to catalogue probe '{probe_info['probe_part_number']}'. "
                f"The probe part number in settings.xml may be incorrect. "
                f"See https://github.com/SpikeInterface/probeinterface/issues/407 for details."
            )


def _annotate_openephys_probe(probe: Probe, probe_info: dict) -> None:
    """
    Annotate a probe with metadata from the parsed settings info.

    Parameters
    ----------
    probe : Probe
        The probe to annotate (modified in place).
    probe_info : dict
        Probe info dict from `_parse_openephys_settings`.
    """
    probe.serial_number = probe_info["serial_number"]
    probe.name = probe_info["name"]
    probe.annotate(part_number=probe_info["probe_part_number"])

    if probe_info["slot"] is not None:
        probe.annotate(slot=probe_info["slot"])
    if probe_info["port"] is not None:
        probe.annotate(port=probe_info["port"])
    if probe_info["dock"] is not None:
        probe.annotate(dock=probe_info["dock"])
    if probe_info.get("settings_channel_keys") is not None:
        settings_channel_keys = probe_info["settings_channel_keys"]
        if probe_info.get("custom_channel_map") is not None:
            settings_channel_keys = np.array(settings_channel_keys)[probe_info["custom_channel_map"]]
        probe.annotate_contacts(settings_channel_key=settings_channel_keys)

    adc_sampling_table = probe.annotations.get("adc_sampling_table")
    _annotate_probe_with_adc_sampling_info(probe, adc_sampling_table)


def read_openephys(
    settings_file: str | Path,
    stream_name: Optional[str] = None,
    probe_name: Optional[str] = None,
    serial_number: Optional[str] = None,
    fix_x_position_for_oe_5: bool = True,
    raise_error: bool = True,
) -> Probe:
    """
    Read a Neuropixels probe geometry from an Open Ephys settings.xml file.

    A single settings.xml can describe multiple probes (one ``<NP_PROBE>`` element
    per probe). When the file contains more than one probe, use one of the three
    mutually exclusive selectors (``stream_name``, ``probe_name``, or
    ``serial_number``) to choose which probe to return.

    In case of a "Channel Map" processor in the signal chain that comes before the "Record Node",
    the probe geometry and settings channel names will be sliced to the order of channels specified
    in the channel map. Therefore, the probe is always wired from 0 to N-1.

    Open Ephys versions 0.5.x, 0.6.x, and 1.0 are supported. For version
    0.6.x+, probe names are inferred from ``<STREAM>`` elements. For version
    0.5.x (no ``<STREAM>`` elements), probes are named sequentially based on
    the processor's ``nodeId`` (e.g. ``"100.0"``, ``"100.1"``).

    Parameters
    ----------
    settings_file : Path or str
        Path to the Open Ephys settings.xml file. Each experiment under a
        Record Node has its own settings file (``settings.xml`` for experiment 1,
        ``settings_2.xml`` for experiment 2, etc.). The caller is responsible
        for passing the correct one.
    stream_name : str or None
        Select a probe by substring match against probe names derived from
        ``<STREAM>`` elements in the settings.xml. For example, if the
        settings file has probes ``"ProbeA"``, ``"ProbeB"``, ``"ProbeC"``,
        any string containing ``"ProbeC"`` as a substring will select that
        probe. This accepts the oebin folder name
        (e.g. ``"Neuropix-PXI-100.ProbeC-AP"``), the short plugin stream
        name (e.g. ``"ProbeC"``), or any other string that contains the
        probe name. Mutually exclusive with ``probe_name`` and
        ``serial_number``.
    probe_name : str or None
        Select a probe by exact match against its name (e.g. ``"ProbeB"``).
        Useful for interactive use. Mutually exclusive with ``stream_name``
        and ``serial_number``.
    serial_number : str or None
        Select a probe by exact match against its serial number. Useful for
        automated pipelines that track probes by hardware serial. Mutually
        exclusive with ``stream_name`` and ``probe_name``.
    fix_x_position_for_oe_5 : bool
        Correct a y-position bug in the Neuropix-PXI plugin for Open Ephys
        < 0.6.0, where multi-shank probe y-coordinates included an erroneous
        shank pitch offset. Despite the parameter name, this corrects the y
        (not x) position. Default True.
    raise_error : bool
        If True, any error raises an exception. If False, None is returned.
        Default True.

    Returns
    -------
    probe : Probe or None
        The wired probe object. Returns None if ``raise_error`` is False and an error occurs.

    Notes
    -----
    Electrode positions are only available when recording with the
    Neuropix-PXI plugin version >= 0.3.3.
    """
    probes_info = _parse_openephys_settings(settings_file, fix_x_position_for_oe_5, raise_error)
    if probes_info is None:
        return None

    probe_info = _select_openephys_probe_info(probes_info, stream_name, probe_name, serial_number, raise_error)
    if probe_info is None:
        return None

    full_probe = build_neuropixels_probe(probe_part_number=probe_info["probe_part_number"])
    probe = _slice_openephys_catalogue_probe(full_probe, probe_info)
    _annotate_openephys_probe(probe, probe_info)

    chans_saved = get_saved_channel_indices_from_openephys_settings(settings_file, stream_name=stream_name)
    if chans_saved is not None:
        probe = probe.get_slice(chans_saved)

    # Wire the probe: in case of a channel map preceding the record node, the probe is already sliced to the custom
    # channel selection, so we can use identity mapping.
    probe.set_device_channel_indices(np.arange(probe.get_contact_count()))
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
