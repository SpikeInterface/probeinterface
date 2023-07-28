"""
Read/write probe info using a variety of formats:
  * probeinterface (.json)
  * PRB (.prb)
  * CSV (.csv)
  * mearec (.h5)
  * spikeglx (.meta)
  * ironclust/jrclust (.mat)
  * Neurodata Without Borders (.nwb)

"""
from pathlib import Path
from typing import Union, Optional, List
import re
import warnings
import json
from collections import OrderedDict
from packaging.version import Version, parse
import numpy as np

from . import __version__
from .probe import Probe
from .probegroup import ProbeGroup
from .utils import import_safely


def _probeinterface_format_check_version(d):
    """
    Check format version of probeinterface JSON file
    """

    pass


def read_probeinterface(file):
    """
    Read probeinterface JSON-based format.

    Parameters
    ----------
    file: Path or str
        The file path

    Returns
    --------
    probegroup : ProbeGroup object

    """

    file = Path(file)
    with open(file, "r", encoding="utf8") as f:
        d = json.load(f)

    # check version
    _probeinterface_format_check_version(d)

    # create probegroup from dict
    return ProbeGroup.from_dict(d)


def write_probeinterface(file: Union[str, Path], probe_or_probegroup: Union[Probe, ProbeGroup]):
    """
    Write a probeinterface JSON file.

    The format handles several probes in one file.

    Parameters
    ----------
    file : Path or str
        The file path
    probe_or_probegroup : Probe or ProbeGroup object
        If probe is given a probegroup is created anyway

    """

    if isinstance(probe_or_probegroup, Probe):
        probe = probe_or_probegroup
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
    elif isinstance(probe_or_probegroup, ProbeGroup):
        probegroup = probe_or_probegroup
    else:
        raise ValueError("write_probeinterface : need probe or probegroup")

    file = Path(file)

    d = OrderedDict()
    d["specification"] = "probeinterface"
    d["version"] = __version__
    d.update(probegroup.to_dict(array_as_list=True))

    with open(file, "w", encoding="utf8") as f:
        json.dump(d, f, indent=4)


tsv_label_map_to_BIDS = {
    "contact_ids": "contact_id",
    "probe_ids": "probe_id",
    "contact_shapes": "contact_shape",
    "shank_ids": "shank_id",
    "si_units": "xyz_units",
}
tsv_label_map_to_probeinterface = {v: k for k, v in tsv_label_map_to_BIDS.items()}


def read_BIDS_probe(folder: Union[str, Path], prefix: Optional[str] = None) -> ProbeGroup:
    """
    Read to BIDS probe format.

    This requires a probes.tsv and a contacts.tsv file
    and potentially corresponding files in JSON format.

    Parameters
    ----------
    folder: Path or str
        The folder to scan for probes and contacts files
    prefix : str
        Prefix of the probes and contacts files

    Returns
    --------
    probegroup : ProbeGroup object

    """

    pd = import_safely("pandas")

    folder = Path(folder)
    probes = {}
    probegroup = ProbeGroup()

    # Identify source files for probes and contacts information
    if prefix is None:
        probes_files = [f for f in folder.iterdir() if f.name.endswith("probes.tsv")]
        contacts_files = [f for f in folder.iterdir() if f.name.endswith("contacts.tsv")]
        if len(probes_files) != 1 or len(contacts_files) != 1:
            raise ValueError("Did not find one probes.tsv and one contacts.tsv file")
        probes_file = probes_files[0]
        contacts_file = contacts_files[0]
    else:
        probes_file = folder / f"{prefix}_probes.tsv"
        contacts_file = folder / f"{prefix}_contacts.tsv"
        for file in [probes_file, contacts_file]:
            if not file.exists():
                raise ValueError(f"Source file does not exist ({file})")

    # Step 1: READING CONTACTS.TSV
    converters = {
        "x": float,
        "y": float,
        "z": float,
        "contact_shapes": str,
        "probe_index": int,
        "probe_id": str,
        "shank_id": str,
        "contact_id": str,
        "radius": float,
        "width": float,
        "height": float,
    }
    df = pd.read_csv(contacts_file, sep="\t", header=0, keep_default_na=False, converters=converters)  #  dtype=str,
    df.replace(to_replace={"n/a": ""}, inplace=True)
    df.rename(columns=tsv_label_map_to_probeinterface, inplace=True)

    if "probe_ids" not in df:
        raise ValueError("probes.tsv file does not contain probe_id column")
    if "contact_ids" not in df:
        raise ValueError("contacts.tsv file does not contain contact_id column")

    for probe_id in df["probe_ids"].unique():
        df_probe = df[df["probe_ids"] == probe_id].copy()

        # adding default values required by probeinterface if not present in
        # source files
        if "contact_shapes" not in df_probe:
            df_probe["contact_shapes"] = "circle"
            df_probe["radius"] = 1
            print(f"There is no contact shape provided for probe {probe_id}, a " f"dummy circle with 1um is created")

        if "x" not in df_probe:
            df_probe["x"] = np.arange(len(df_probe.index), dtype=float)
            print(
                f"There is no x coordinate provided for probe {probe_id}, a " f"dummy linear x coordinate is created."
            )

        if "y" not in df_probe:
            df_probe["y"] = 0.0
            print(
                f"There is no y coordinate provided for probe {probe_id}, a " f"dummy constant y coordinate is created."
            )

        if "si_units" not in df_probe:
            df_probe["si_units"] = "um"
            print(f"There is no SI units provided for probe {probe_id}, a " f"dummy SI unit (um) is created.")

        # create probe object and register with probegroup
        probe = Probe.from_dataframe(df=df_probe)
        probe.annotate(probe_id=probe_id)

        probes[str(probe_id)] = probe
        probegroup.add_probe(probe)

        ignore_annotations = [
            "probe_ids",
            "contact_ids",
            "contact_shapes",
            "x",
            "y",
            "z",
            "shank_ids",
            "si_units",
            "device_channel_indices",
            "radius",
            "width",
            "height",
            "probe_num",
            "device_channel_indices",
        ]
        df_others = df_probe.drop(ignore_annotations, axis=1, errors="ignore")
        for col_name in df_others.columns:
            probe.annotate(**{col_name: df_probe[col_name].values})

    # Step 2: READING PROBES.TSV
    df = pd.read_csv(probes_file, sep="\t", header=0, keep_default_na=False, dtype=str)
    df.replace(to_replace={"n/a": ""}, inplace=True)

    if "probe_id" not in df:
        raise ValueError(f"{probes_file} file does not contain probe_id column")

    for row_idx, row in df.iterrows():
        probe_id = row["probe_id"]
        if probe_id not in probes:
            print(
                f"Probe with id {probe_id} is present in probes.tsv but not "
                f"in contacts.tsv file. Ignoring entry in probes.tsv."
            )
            continue

        probe = probes[probe_id]
        probe.annotate(**dict(row.items()))

        # for string based annotations use '' instead of None as default
        for string_annotation in ["name", "manufacturer"]:
            if probe.annotations.get(string_annotation, None) is None:
                probe.annotations[string_annotation] = ""

    # Step 3: READING PROBES.JSON (optional)
    probes_dict = {}
    probe_json = probes_file.with_suffix(".json")
    if probe_json.exists():
        with open(probe_json, "r") as f:
            probes_dict = json.load(f)

    if "ProbeId" in probes_dict:
        for probe_id, probe_info in probes_dict["ProbeId"].items():
            probe = probes[probe_id]
            for probe_param, param_value in probe_info.items():
                if probe_param == "contour":
                    probe.probe_planar_contour = np.array(param_value)

                elif probe_param == "units":
                    if probe.si_units is None:
                        probe.si_units = param_value
                    elif probe.si_units != param_value:
                        raise ValueError(f"Inconsistent si_units for probe " f"{probe_id}")
                else:
                    probe.annotate(**{probe_param: param_value})

    # Step 4: READING CONTACTS.JSON (optional)
    contacts_dict = {}
    contact_json = contacts_file.with_suffix(".json")
    if contact_json.exists():
        with open(contact_json, "r") as f:
            contacts_dict = json.load(f)

    if "ContactId" in contacts_dict:
        # collect all contact parameters used in this file
        contact_params = [k for v in contacts_dict["ContactId"].values() for k in v.keys()]
        contact_params = np.unique(contact_params)

        # collect contact information for each probe_id
        for probe in probes.values():
            contact_ids = probe.contact_ids
            for contact_param in contact_params:
                # collect parameters across contact ids to add to probe
                value_list = [contacts_dict["ContactId"][str(c)].get(contact_param, None) for c in contact_ids]

                probe.annotate(**{contact_param: value_list})

    return probegroup


def write_BIDS_probe(folder: Union[str, Path], probe_or_probegroup: Union[Probe, ProbeGroup], prefix=""):
    """
    Write to probe and contact formats as proposed
    for ephy BIDS extension (tsv & json based).

    The format handles several probes in one file.

    Parameters
    ----------
    folder : Path or str
        The folder name.
    probe_or_probegroup : Probe or ProbeGroup
        If probe is given a probegroup is created anyway.
    prefix : str
        A prefix to be added to the filenames

    """

    pd = import_safely("pandas")

    if isinstance(probe_or_probegroup, Probe):
        probe = probe_or_probegroup
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
    elif isinstance(probe_or_probegroup, ProbeGroup):
        probegroup = probe_or_probegroup
    else:
        raise ValueError("probe_or_probegroup has to be" "of type Probe or ProbeGroup")
    folder = Path(folder)

    # ensure that prefix and file type indicator are separated by an underscore
    if prefix != "" and prefix[-1] != "_":
        prefix = prefix + "_"

    probes = probegroup.probes

    # Step 1: GENERATION OF PROBE.TSV
    # ensure required keys (probe_id, probe_type) are present

    if any("probe_id" not in p.annotations for p in probes):
        probegroup.auto_generate_probe_ids()

    for probe in probes:
        if "probe_id" not in probe.annotations:
            raise ValueError(
                "Export to BIDS probe format requires "
                "the probe id to be specified as an annotation "
                "(probe_id). You can do this via "
                "`probegroup.auto_generate_ids."
            )
        if "type" not in probe.annotations:
            raise ValueError(
                "Export to BIDS probe format requires " "the probe type to be specified as an " "annotation (type)"
            )

    # extract all used annotation keys
    keys_by_probe = [list(p.annotations) for p in probes]
    keys_concatenated = np.concatenate(keys_by_probe)
    annotation_keys = np.unique(keys_concatenated)

    # generate a tsv table capturing probe information
    index = range(len([p.annotations["probe_id"] for p in probes]))
    df = pd.DataFrame(index=index)
    for annotation_key in annotation_keys:
        df[annotation_key] = [p.annotations[annotation_key] for p in probes]
    df["n_shanks"] = [len(np.unique(p.shank_ids)) for p in probes]

    # Note: in principle it would also be possible to add the probe width and
    # depth here based on the probe contour information. However this would
    # require an alignment of the probe within the coordinate system.

    # substitute empty values by BIDS default and create tsv file
    df.fillna("n/a", inplace=True)
    df.replace(to_replace="", value="n/a", inplace=True)
    df.to_csv(folder.joinpath(prefix + "probes.tsv"), sep="\t", index=False)

    # Step 2: GENERATION OF PROBE.JSON
    probes_dict = {}
    for probe in probes:
        probe_id = probe.annotations["probe_id"]
        probes_dict[probe_id] = {
            "contour": probe.probe_planar_contour.tolist(),
            "units": probe.si_units,
        }
        probes_dict[probe_id].update(probe.annotations)

    with open(folder.joinpath(prefix + "probes.json"), "w", encoding="utf8") as f:
        json.dump({"ProbeId": probes_dict}, f, indent=4)

    # Step 3: GENERATION OF CONTACTS.TSV
    # ensure required contact identifiers are present
    for probe in probes:
        if probe.contact_ids is None:
            raise ValueError(
                "Contacts must have unique contact ids "
                "and not None for export to BIDS probe format."
                "Use `probegroup.auto_generate_contact_ids`."
            )

    df = probegroup.to_dataframe()
    index = range(sum([p.get_contact_count() for p in probes]))
    df.rename(columns=tsv_label_map_to_BIDS, inplace=True)

    df["probe_id"] = [p.annotations["probe_id"] for p in probes for _ in p.contact_ids]
    df["coordinate_system"] = ["relative cartesian"] * len(index)

    channel_indices = []
    for probe in probes:
        if probe.device_channel_indices:
            channel_indices.extend(probe.device_channel_indices)
        else:
            channel_indices.extend([-1] * probe.get_contact_count())
    df["device_channel_indices"] = channel_indices

    df.fillna("n/a", inplace=True)
    df.replace(to_replace="", value="n/a", inplace=True)
    df.to_csv(folder.joinpath(prefix + "contacts.tsv"), sep="\t", index=False)

    # Step 4: GENERATING CONTACTS.JSON
    contacts_dict = {}
    for probe in probes:
        for cidx, contact_id in enumerate(probe.contact_ids):
            cdict = {"contact_plane_axes": probe.contact_plane_axes[cidx].tolist()}
            contacts_dict[contact_id] = cdict

    with open(folder.joinpath(prefix + "contacts.json"), "w", encoding="utf8") as f:
        json.dump({"ContactId": contacts_dict}, f, indent=4)


def read_prb(file: Union[str, Path]) -> ProbeGroup:
    """
    Read a PRB file and return a ProbeGroup object.

    Since PRB does not handle contact shapes, contacts are set to be circle of 5um radius.
    Same for the probe shape, where an auto shape is created.

    PRB format does not contain any information about the channel of the probe
    Only the channel index on device is given.

    Parameters
    ----------
    file : Path or str
        The file path

    Returns
    --------
    probegroup : ProbeGroup object
    """

    file = Path(file).absolute()
    assert file.is_file()
    with file.open("r") as f:
        contents = f.read()
    contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
    prb = {}
    exec(contents, None, prb)
    prb = {k.lower(): v for (k, v) in prb.items()}

    if "channel_groups" not in prb:
        raise ValueError("This file is not a standard PRB file")

    probegroup = ProbeGroup()
    for i, group in prb["channel_groups"].items():
        probe = Probe(ndim=2, si_units="um")

        chans = np.array(group["channels"], dtype="int64")
        positions = np.array([group["geometry"][c] for c in chans], dtype="float64")

        probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
        probe.create_auto_shape(probe_type="tip")

        probe.set_device_channel_indices(chans)
        probegroup.add_probe(probe)

    return probegroup


def read_maxwell(file: Union[str, Path], well_name: str = "well000", rec_name: str = "rec0000") -> Probe:
    """
    Read a maxwell file and return a Probe object. The Maxwell file format can be
    either Maxone (and thus just the file name is needed), or MaxTwo. In case
    of the latter, you need to explicitly specify what is the well number of
    interest (well000 by default), and the recording session (since there can
    be several. Default is rec0000)

    Parameters
    ----------
    file : Path or str
        The file name

    well_name : str
        If MaxTwo file format, the well_name to extract the mapping from
        (default is well000)

    rec_name : str
        If MaxTwo file format, the recording session to extract the mapping
        from (default is rec0000)

    Returns
    --------
    probe : Probe object

    """

    file = Path(file).absolute()
    assert file.is_file()

    h5py = import_safely("h5py")
    my_file = h5py.File(file, mode="r")

    if "mapping" in my_file.keys():
        mapping = my_file["mapping"][:]
    else:
        mapping = my_file["wells"][well_name][rec_name]["settings"]["mapping"][:]

    prb = {"channel_groups": {1: {}}}

    channels = list(mapping["channel"])
    electrodes = list(mapping["electrode"])
    x_pos = list(mapping["x"])
    y_pos = list(mapping["y"])
    geometry = {}
    for c, x, y in zip(channels, x_pos, y_pos):
        geometry[c] = [x, y]

    my_file.close()

    prb["channel_groups"][1]["geometry"] = geometry
    prb["channel_groups"][1]["channels"] = channels

    probe = Probe(ndim=2, si_units="um")

    chans = np.array(prb["channel_groups"][1]["channels"], dtype="int64")
    positions = np.array([prb["channel_groups"][1]["geometry"][c] for c in chans], dtype="float64")

    probe.set_contacts(positions=positions, shapes="rect", shape_params={"width": 5.45, "height": 9.3})
    probe.annotate_contacts(electrode=electrodes)
    probe.set_planar_contour(([-12.5, -12.5], [3845, -12.5], [3845, 2095], [-12.5, 2095]))

    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def read_3brain(file: Union[str, Path], mea_pitch: float = 42, electrode_width: float = 21) -> Probe:
    """
    Read a 3brain file and return a Probe object. The 3brain file format can be
    either an .h5 file or a .brw

    Parameters
    ----------
    file : Path or str
        The file name

    mea_pitch : float
        The inter-electrode distance (pitch) between electrodes

    electrode_width : float
        Width of the electrodes in um

    Returns
    --------
    probe : Probe object

    """
    file = Path(file).absolute()
    assert file.is_file()

    h5py = import_safely("h5py")
    rf = h5py.File(file, "r")

    # get channel positions
    channels = rf["3BRecInfo/3BMeaStreams/Raw/Chs"][:]
    rows = channels["Row"] - 1
    cols = channels["Col"] - 1
    positions = np.vstack((rows, cols)).T * mea_pitch

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="square", shape_params={"width": electrode_width})
    probe.annotate_contacts(row=rows)
    probe.annotate_contacts(col=cols)
    probe.create_auto_shape(probe_type="rect", margin=mea_pitch)
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def write_prb(file, probegroup, total_nb_channels=None, radius=None, group_mode="by_probe"):
    """
    Write ProbeGroup into a prb file.

    This format handles:
      * multi Probe with channel group index key
      * channel positions with "geometry"
      * device_channel_indices with "channels "key

    Note: much information is lost in the PRB format:
      * contact shape
      * shape
      * channel index

    Note:
      * "total_nb_channels" is needed by spyking-circus
      * "radius" is needed by spyking-circus
      * "graph" is not handled

    """
    assert group_mode in ("by_probe", "by_shank")

    if len(probegroup.probes) == 0:
        raise ValueError("Bad boy")

    for probe in probegroup.probes:
        if probe.device_channel_indices is None:
            raise ValueError("For PRB format device_channel_indices must be set")

    with open(file, "w") as f:
        if total_nb_channels is not None:
            f.write(f"total_nb_channels = {total_nb_channels}\n")
        if radius is not None:
            f.write(f"radius = {radius}\n")

        f.write("channel_groups = {\n")

        if group_mode == "by_probe":
            loop = enumerate(probegroup.probes)
        elif group_mode == "by_shank":
            shanks = []
            for probe in probegroup.probes:
                shanks.extend(probe.get_shanks())
            loop = enumerate(shanks)

        for group_ind, probe_or_shank in loop:
            f.write(f"    {group_ind}:\n")
            f.write("        {\n")
            channels = probe_or_shank.device_channel_indices
            keep = channels >= 0
            #  channels -1 are not wired

            chans = list(channels[keep])
            f.write(f"           'channels': {chans},\n")
            f.write("           'geometry':  {\n")
            for c in range(probe_or_shank.get_contact_count()):
                if not keep[c]:
                    continue
                pos = list(probe_or_shank.contact_positions[c, :])
                f.write(f"               {channels[c]}: {pos},\n")
            f.write("           }\n")
            f.write("       },\n")

        f.write("}\n")


def read_csv(file):
    """
    Return a 2 or 3 columns csv file with contact positions
    """

    raise NotImplementedError


def write_csv(file, probe):
    """
    Write contact postions into a 2 or 3 columns csv file
    """

    raise NotImplementedError


polygon_description = {
    "default": [
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

# A map from probe type to geometry_parameters
npx_probe = {
    # Neuropixels 1.0
    # This probably should be None or something else because NOT ONLY the neuropixels 1.0 have that imDatPrb_type
    0: {
        "probe_name": "Neuropixels 1.0",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 16,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["default"],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 2.0 - Single Shank
    21: {
        "probe_name": "Neuropixels 2.0 - Single Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["default"],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Four Shank
    24: {
        "probe_name": "Neuropixels 2.0 - Four Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncol": 2,
        "polygon": polygon_description["default"],
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
        "probe_name": "Phase3a",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 16.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["default"],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
        ),
    },
    # Neuropixels 1.0-NHP Short (10mm)
    1015: {
        "probe_name": "Neuropixels 1.0-NHP - short",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["default"],
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
    1022: {
        "probe_name": "Neuropixels 1.0-NHP - medium",
        "x_pitch": 103,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["nhp125"],
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
    1030: {
        "probe_name": "Neuropixels 1.0-NHP - long SOI90 staggered",
        "x_pitch": 56,
        "y_pitch": 20,
        "stagger": 12,
        "contact_width": 12,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["nhp90"],
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
    1031: {
        "probe_name": "Neuropixels 1.0-NHP - long SOI125 staggered",
        "x_pitch": 91,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["nhp125"],
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
    1032: {
        "probe_name": "Neuropixels 1.0-NHP - long SOI125 linear",
        "x_pitch": 103,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 2,
        "polygon": polygon_description["nhp125"],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Ultra probe
    1100: {
        "probe_name": "Ultra probe",
        "x_pitch": 6,
        "y_pitch": 6,
        "contact_width": 5,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncol": 8,
        "polygon": polygon_description["default"],
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
probe_number_to_probe_type = {
    "PRB_1_4_0480_1": 0,
    "PRB_1_4_0480_1_C": 0,
    "NP1010": 0,
    "NP1015": 1015,
    "NP1022": 1022,
    "NP1030": 1030,
    "NP1031": 1031,
    "NP1032": 1032,
    None: 0,
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
    if len(imro_table_header) == 3:
        # In older versions of neuropixel arrays (phase 3A), imro tables were structured differently.
        probe_serial_number, probe_option, num_contact = imro_table_header
        imDatPrb_type = "Phase3a"
    elif len(imro_table_header) == 2:
        imDatPrb_type, num_contact = imro_table_header
    else:
        raise ValueError(f"read_imro error, the header has a strange length: {imro_table_header}")

    if imDatPrb_type in [0, None]:
        imDatPrb_type = probe_number_to_probe_type[imDatPrb_pn]

    probe_description = npx_probe[imDatPrb_type]
    fields = probe_description["fields_in_imro_table"]
    contact_info = {k: [] for k in fields}
    for field_values_str in imro_table_values_list:  # Imro table values look like '(value, value, value, ... '
        values = tuple(map(int, field_values_str[1:].split(" ")))
        # Split them by space to get (int('value'), int('value'), int('value'), ...)
        for field, field_value in zip(fields, values):
            contact_info[field].append(field_value)

    channel_ids = np.array(contact_info["channel_ids"])
    if "elect_ids" in contact_info:
        elec_ids = np.array(contact_info["elect_ids"])
    else:
        banks = np.array(contact_info["banks"])
        elec_ids = banks * 384 + channel_ids

    # compute position
    y_idx, x_idx = np.divmod(elec_ids, probe_description["ncol"])
    x_pitch = probe_description["x_pitch"]
    y_pitch = probe_description["y_pitch"]

    stagger = np.mod(y_idx + 1, 2) * probe_description["stagger"]
    x_pos = x_idx * x_pitch + stagger
    y_pos = y_idx * y_pitch
    positions = np.stack((x_pos, y_pos), axis=1)

    if imDatPrb_type == 24:
        shank_ids = np.array(contact_info["shank_id"])
        contact_ids = [f"s{shank_id}e{elec_id}" for shank_id, elec_id in zip(shank_ids, elec_ids)]
    else:
        shank_ids = None
        contact_ids = [f"e{elec_id}" for elec_id in elec_ids]

    # construct Probe object
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shank_ids=shank_ids,
        shape_params={"width": probe_description["contact_width"]},
    )

    probe.set_contact_ids(contact_ids)

    # Add planar contour
    polygon = np.array(probe_description["polygon"])
    contour = []
    shank_pitch = probe_description["shank_pitch"]
    for shank_id in range(probe_description["shank_number"]):
        shift = [shank_pitch * shank_id, 0]
        contour += list(polygon + shift)

    # shift
    contour = np.array(contour) - [11, 11]
    probe.set_planar_contour(contour)

    # this is scalar annotations
    probe_name = probe_description["probe_name"]
    probe.annotate(
        name=probe_name,
        manufacturer="IMEC",
        probe_type=imDatPrb_type,
    )

    # this is vector annotations
    vector_properties = ("channel_ids", "banks", "references", "ap_gains", "lf_gains", "ap_hp_filters")
    vector_properties_available = {k: v for k, v in contact_info.items() if k in vector_properties}
    probe.annotate_contacts(**vector_properties_available)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def write_imro(file, probe):
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

    if probe_type == 0:
        for ch in range(len(data)):
            ret.append(
                f"({ch} 0 {annotations['references'][ch]} {annotations['ap_gains'][ch]} "
                f"{annotations['lf_gains'][ch]} {annotations['ap_hp_filters'][ch]})"
            )

    elif probe_type == 21:
        for ch in range(len(data)):
            ret.append(
                f"({data['device_channel_indices'][ch]} {annotations['banks'][ch]} "
                f"{annotations['references'][ch]} {data['contact_ids'][ch][1:]})"
            )

    elif probe_type == 24:
        for ch in range(len(data)):
            ret.append(
                f"({data['device_channel_indices'][ch]} {data['shank_ids'][ch]} {annotations['banks'][ch]} "
                f"{annotations['references'][ch]} {data['contact_ids'][ch][3:]})"
            )
    else:
        raise ValueError(f"unknown imro type : {probe_type}")
    with open(file, "w") as f:
        f.write("".join(ret))


def read_spikeglx(file: Union[str, Path]) -> Probe:
    """
    Read probe position for the meta file generated by SpikeGLX

    See http://billkarsh.github.io/SpikeGLX/#metadata-guides for implementation.
    The x_pitch/y_pitch/width are set automatically depending the NP version.

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
    imDatPrb_pn = meta.get("imDatPrb_pn", None)

    probe = _read_imro_string(imro_str=imro_table, imDatPrb_pn=imDatPrb_pn)

    # sometimes we need to slice the probe when not all channels are saved
    saved_chans = get_saved_channel_indices_from_spikeglx_meta(meta_file)
    # remove the SYS chans
    saved_chans = saved_chans[saved_chans < probe.get_contact_count()]
    if saved_chans.size != probe.get_contact_count():
        # slice if needed
        probe = probe.get_slice(saved_chans)

    return probe


def parse_spikeglx_meta(meta_file: Union[str, Path]) -> dict:
    """
    Parse the "meta" file from spikeglx into a dict.
    All fiields are kept in txt format and must also parsed themself.
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


def get_saved_channel_indices_from_spikeglx_meta(meta_file: Union[str, Path]) -> np.array:
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


def read_openephys(
    settings_file: Union[str, Path],
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
    signal_chain = root.find("SIGNALCHAIN")
    neuropix_pxi = None
    record_node = None
    for processor in signal_chain:
        if "PROCESSOR" == processor.tag:
            name = processor.attrib["name"]
            if "Neuropix-PXI" in name:
                neuropix_pxi = processor
            if "Record Node" in name:
                record_node = processor

    if neuropix_pxi is None:
        if raise_error:
            raise Exception("Open Ephys can only be read when the Neuropix-PXI plugin is used")
        return None

    if "NodeId" in neuropix_pxi.attrib:
        node_id = neuropix_pxi.attrib["NodeId"]
    elif "nodeId" in neuropix_pxi.attrib:
        node_id = neuropix_pxi.attrib["nodeId"]
    else:
        node_id = None
    neuropix_pxi_version = parse(neuropix_pxi.attrib["libraryVersion"])
    if neuropix_pxi_version < parse("0.3.3"):
        if raise_error:
            raise Exception("Electrode locations are available from Neuropix-PXI version 0.3.3")
        return None

    # read STREAM fields if present (>=0.6.x)
    stream_fields = neuropix_pxi.findall("STREAM")
    if len(stream_fields) > 0:
        has_streams = True
        streams = []
        for stream_field in stream_fields:
            streams.append(stream_field.attrib["name"])
        probe_names_used = np.unique([stream.split("-")[0] for stream in streams])
    else:
        has_streams = False
        probe_names_used = None

    editor = neuropix_pxi.find("EDITOR")
    np_probes = editor.findall("NP_PROBE")

    if len(np_probes) == 0:
        if raise_error:
            raise Exception("NP_PROBE field not found in settings")
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
        np_serial_number = np_probe.attrib["probe_serial_number"]
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
            shank_ids = np.array([int(val[val.find(":") + 1 :]) for val in channel_values])
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

        contact_ids = []
        pname = np_probe.attrib["probe_name"]
        if "2.0" in pname:
            x_shift = -8
            if "Multishank" in pname:
                ptype = 24
            else:
                ptype = 21
        elif "NHP" in pname:
            ptype = 0
            x_shift = -11
        elif "1.0" in pname:
            ptype = 0
            x_shift = -11
        elif "Ultra" in pname:
            ptype = 1100
            x_shift = -8
        else:  # Probe type unknown
            ptype = None
            x_shift = 0

        if fix_x_position_for_oe_5 and oe_version < parse("0.6.0") and shank_ids is not None:
            positions[:, 1] = positions[:, 1] - npx_probe[ptype]["shank_pitch"] * shank_ids

        # x offset
        positions[:, 0] += x_shift

        for i, pos in enumerate(positions):
            if ptype is None:
                contact_ids = None
                break

            stagger = np.mod(pos[1] / npx_probe[ptype]["y_pitch"] + 1, 2) * npx_probe[ptype]["stagger"]
            shank_id = shank_ids[0] if ptype == 24 else 0

            contact_id = int(
                (pos[0] - stagger - npx_probe[ptype]["shank_pitch"] * shank_id) / npx_probe[ptype]["x_pitch"]
                + npx_probe[ptype]["ncol"] * pos[1] / npx_probe[ptype]["y_pitch"]
            )
            if ptype == 24:
                contact_ids.append(f"s{shank_id}e{contact_id}")
            else:
                contact_ids.append(f"e{contact_id}")

        np_probe_dict = {
            "channel_names": channel_names,
            "shank_ids": shank_ids,
            "contact_ids": contact_ids,
            "positions": positions,
            "slot": slot,
            "port": port,
            "dock": dock,
            "serial_number": np_serial_number,
            "ptype": ptype,
        }
        # Sequentially assign probe names
        np_probe_dict.update({"name": probe_names_used[probe_idx]})
        np_probes_info.append(np_probe_dict)

    # now select correct probe (if multiple)
    if len(np_probes) > 1:
        found = False

        if stream_name is not None:
            assert probe_name is None and serial_number is None, (
                "Use one of 'stream_name', 'probe_name', " "or 'serial_number'"
            )
            for probe_idx, probe_info in enumerate(np_probes_info):
                if probe_info["name"] in stream_name:
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
                "More than one probe found. Use one of 'stream_name', 'probe_name', or 'serial_number' "
                "to select the right probe"
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
                        f"Inconsistency betweem provided stream {stream_name} and available probe "
                        f"{available_probe_name}"
                    )
                return None
        if probe_name:
            if probe_name != available_probe_name:
                if raise_error:
                    raise Exception(
                        f"Inconsistency betweem provided probe name {probe_name} and available probe "
                        f"{available_probe_name}"
                    )
                return None
        if serial_number:
            if str(serial_number) != available_serial_number:
                if raise_error:
                    raise Exception(
                        f"Inconsistency betweem provided serial number {serial_number} and available serial numbers "
                        f"{available_serial_number}"
                    )
                return None
        probe_idx = 0

    np_probe_info = np_probes_info[probe_idx]
    np_probe = np_probes[probe_idx]
    positions = np_probe_info["positions"]
    shank_ids = np_probe_info["shank_ids"]
    pname = np_probe.attrib["probe_name"]

    ptype = np_probe_info["ptype"]
    if ptype in npx_probe:
        contact_width = npx_probe[ptype]["contact_width"]
        shank_pitch = npx_probe[ptype]["shank_pitch"]
    else:
        contact_width = 12
        shank_pitch = 250

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

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shank_ids=shank_ids,
        shape_params={"width": contact_width},
    )
    probe.annotate(
        name=pname,
        manufacturer="IMEC",
        probe_name=pname,
        probe_part_number=np_probe.attrib["probe_part_number"],
        probe_serial_number=np_probe.attrib["probe_serial_number"],
    )

    if contact_ids is not None:
        probe.set_contact_ids(contact_ids)

    polygon = polygon_description["default"]
    if shank_ids is None:
        contour = polygon
    else:
        contour = []
        for i, shank_id in enumerate(np.unique(shank_ids)):
            contour += list(np.array(polygon) + [shank_pitch * i, 0])

    # shift
    contour = np.array(contour) - [11, 11]
    probe.set_planar_contour(contour)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def get_saved_channel_indices_from_openephys_settings(settings_file, stream_name):
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
                    if rs is not None and rs != "ALL":
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
                        if recording_state != "ALL":
                            chans_saved = np.array([chan for chan, r in enumerate(recording_state) if int(r) == 1])
    return chans_saved


def read_mearec(file: Union[str, Path]) -> Probe:
    """
    Read probe position, and contact shape from a MEArec file.

    See https://mearec.readthedocs.io/en/latest/ and https://doi.org/10.1007/s12021-020-09467-7 for implementation.

    Parameters
    ----------
    file : Path or str
        The file path

    Returns
    -------
    probe : Probe object

    """

    file = Path(file).absolute()
    assert file.is_file()

    h5py = import_safely("h5py")

    f = h5py.File(file, "r")
    positions = f["channel_positions"][()]
    electrodes_info = f["info"]["electrodes"]
    electrodes_info_keys = electrodes_info.keys()

    mearec_description = None
    mearec_name = None
    if "electrode_name" in electrodes_info_keys:
        mearec_name = electrodes_info["electrode_name"][()]
        mearec_name = mearec_name.decode("utf-8") if isinstance(mearec_name, bytes) else mearec_name

    if "description" in electrodes_info_keys:
        description = electrodes_info["description"][()]
        mearec_description = description.decode("utf-8") if isinstance(description, bytes) else description

    probe = Probe(ndim=2, si_units="um")

    plane = "yz"  # default
    if "plane" in electrodes_info_keys:
        plane = electrodes_info["plane"][()]
        plane = plane.decode("utf-8") if isinstance(plane, bytes) else plane

    plane_to_columns = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
    columns = plane_to_columns[plane]
    positions_2d = positions[()][:, columns]

    shape = None
    if "shape" in electrodes_info_keys:
        shape = electrodes_info["shape"][()]
        shape = shape.decode("utf-8") if isinstance(shape, bytes) else shape

    size = None
    if "shape" in electrodes_info_keys:
        size = electrodes_info["size"][()]

    shape_params = {}
    if shape is not None and size is not None:
        if shape == "circle":
            shape_params = {"radius": size}
        elif shape == "square":
            shape_params = {"width": 2 * size}
        elif shape == "rect":
            shape_params = {{"width": 2 * size[0], "height": 2 * size[1]}}

    # create contacts
    probe.set_contacts(positions_2d, shapes=shape, shape_params=shape_params)

    # add MEArec annotations
    annotations = dict(mearec_name=mearec_name, mearec_description=mearec_description)
    probe.annotate(**annotations)

    # set device indices
    if electrodes_info["sortlist"][()] not in (b"null", "null"):
        channel_indices = electrodes_info["sortlist"][()]
    else:
        channel_indices = np.arange(positions.shape[0], dtype="int64")
    probe.set_device_channel_indices(channel_indices)

    # create auto shape
    probe.create_auto_shape(probe_type="tip")

    return probe


def read_nwb(nwbfile: Union[Path, str]) -> List[Probe]:
    """
    Load ndx_probeinterface.Probe as
    probeinterface.Probe objects from an NWB file

    Parameters
    ----------
    nwbfile : Path or str
        The path to nwbfile

    Returns
    -------
    probe : List[Probe]
        List of Probe objects
    """
    try:
        import ndx_probeinterface
        import pynwb
    except ImportError:
        raise ImportError("Missing `ndx_probeinterface` or `pynwb`")

    with pynwb.NWBHDF5IO(nwbfile, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        ndx_probes = []
        for device in nwbf.devices:
            if isinstance(device, ndx_probeinterface.Probe):
                ndx_probes.append(device)
        if not ndx_probes:
            core_probe = _from_nwb_ElectrodeTable(nwbf.electrodes)
            return [core_probe]
    probes = []
    for ndx_probe in ndx_probes:
        probes.append(ndx_probeinterface.to_probeinterface(ndx_probe))

    return probes


def _from_nwb_ElectrodeTable(nwbf_electrodes):
    # infer dimension by
    # 1. checking for columns with names 'rel_x', 'rel_y', 'rel_z'
    # 2. checking how many of these columns have elements that are not all zero
    rel_present = [rel for rel in ["rel_x", "rel_y", "rel_z"] if rel in nwbf_electrodes.columns]
    true_rel_present = []
    for rel in rel_present:
        if (nwbf_electrodes[rel][:] == 0).all() == False:
            true_rel_present.append(rel)
    ndim = len(true_rel_present)
    assert ndim >= 1, "Insufficient position information to generate Probe object."

    # no way to read units when only ElectrodeTable is present; just assume microns
    unit = "um"

    # create Probe
    probeinterface_probe = Probe(ndim=ndim, si_units=unit)

    # infer positions
    n_contacts = len(nwbf_electrodes)
    positions = np.zeros((n_contacts, ndim))
    for i in range(ndim):
        positions[:, i] = nwbf_electrodes[true_rel_present[i]][:]

    # set contacts
    probeinterface_probe.set_contacts(
        positions=positions,
    )
    return probeinterface_probe
