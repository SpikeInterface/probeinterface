"""
Read/write probe info using a variety of formats:
  * probeinterface (.json)
  * PRB (.prb)
  * CSV (.csv)
  * mearec (.h5)
  * ironclust/jrclust (.mat)
  * Neurodata Without Borders (.nwb)

"""

from pathlib import Path
import re
import warnings
import json
from collections import OrderedDict
from packaging.version import parse
import numpy as np
from xml.etree import ElementTree

from . import __version__
from .probe import Probe
from .probegroup import ProbeGroup
from .neuropixels_tools import build_neuropixels_probe
from .utils import import_safely


def _probeinterface_format_check_version(d):
    """
    Check format version of probeinterface JSON file
    """

    pass


def read_probeinterface(file: str | Path) -> ProbeGroup:
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


def write_probeinterface(file: str | Path, probe_or_probegroup: Probe | ProbeGroup):
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
        raise TypeError(
            f"write_probeinterface : needs a probe or probegroup you "
            f"entered an object of type: {type(probe_or_probegroup)}"
        )

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


def read_BIDS_probe(folder: str | Path, prefix: str | None = None) -> ProbeGroup:
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
            print(
                f"There is no contact shape provided for probe {probe_id}, a "
                f"dummy circle with 1um radius will be used."
            )

        if "x" not in df_probe:
            df_probe["x"] = np.arange(len(df_probe.index), dtype=float)
            print(
                f"There is no x coordinate provided for probe {probe_id}, a " f"dummy linear x coordinate will be used."
            )

        if "y" not in df_probe:
            df_probe["y"] = 0.0
            print(
                f"There is no y coordinate provided for probe {probe_id}, a "
                f"dummy constant y coordinate will be used."
            )

        if "si_units" not in df_probe:
            df_probe["si_units"] = "um"
            print(f"There is no SI unit provided for probe {probe_id}, a " f"dummy SI unit (um) will be used")

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


def write_BIDS_probe(folder: str | Path, probe_or_probegroup: Probe | ProbeGroup, prefix: str = ""):
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
        raise TypeError(
            f"probe_or_probegroup has to be" "of type Probe or ProbeGroup " f"not type: {type(probe_or_probegroup)}"
        )
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


def read_prb(file: str | Path) -> ProbeGroup:
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
    assert file.is_file(), "'file given is not of type file"
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


def read_maxwell(file: str | Path, well_name: str = "well000", rec_name: str = "rec0000") -> Probe:
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

    probe = Probe(ndim=2, si_units="um", manufacturer="Maxwell Biosystems")

    chans = np.array(prb["channel_groups"][1]["channels"], dtype="int64")
    positions = np.array([prb["channel_groups"][1]["geometry"][c] for c in chans], dtype="float64")

    probe.set_contacts(positions=positions, shapes="rect", shape_params={"width": 5.45, "height": 9.3})
    probe.annotate_contacts(electrode=electrodes)
    probe.set_planar_contour(([-12.5, -12.5], [3845, -12.5], [3845, 2095], [-12.5, 2095]))

    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def read_3brain(file: str | Path, mea_pitch: float = None, electrode_width: float = None) -> Probe:
    """
    Read a 3brain file and return a Probe object. The 3brain file format can be
    either an .h5 file or a .brw

    Parameters
    ----------
    file : Path or str
        The file name

    mea_pitch : float
        The inter-electrode distance (pitch) between electrodes in um, if
        `None` it is tried to be inferred from the chip model in the file or
        set to 42

    electrode_width : float
        The width of the electrodes in um, if `None` it is tried to be inferred
        from the chip model in the file or set to 21

    Returns
    --------
    probe : Probe object

    Notes
    -----
    In case of multiple wells, the function will return the probe of the first
    plate.

    """
    file = Path(file).absolute()
    assert file.is_file()

    h5py = import_safely("h5py")
    rf = h5py.File(file, "r")
    if "3BRecInfo" in rf.keys():  # brw v3.x
        # get channel positions
        channels = rf["3BRecInfo/3BMeaStreams/Raw/Chs"][:]
        rows = channels["Row"] - 1
        cols = channels["Col"] - 1
        if mea_pitch is None:
            mea_pitch = 42
        if electrode_width is None:
            electrode_width = 21
    else:  # brw v4.x
        num_channels = None
        for key in rf:
            if key.startswith("Well_"):
                num_channels = len(rf[key]["StoredChIdxs"])
                break
        assert num_channels is not None, "No Well found in the file"

        num_channels_x = num_channels_y = int(np.sqrt(num_channels))
        assert num_channels_x * num_channels_y == num_channels, (
            "Electrode configuration is not a square. Cannot determine "
            f"configuration of the MEA plate with {num_channels} channels."
        )
        rows = np.repeat(range(num_channels_x), num_channels_y)
        cols = np.tile(range(num_channels_y), num_channels_x)
        if mea_pitch is None or electrode_width is None:
            experiment_settings = json.JSONDecoder().decode(rf["ExperimentSettings"][0].decode())
            model = experiment_settings["MeaPlate"]["Model"].lower()
            # see https://www.3brain.com/products/single-well/hd-mea
            # see https://www.3brain.com/products/multiwell/coreplate-multiwell
            if mea_pitch is None:
                if model.startswith("accura") or model.startswith("coreplate"):
                    mea_pitch = 60
                elif model.startswith("stimulo"):
                    mea_pitch = 81
                else:  # Arena, Prime
                    mea_pitch = 42
            if electrode_width is None:
                if model.startswith("coreplate"):
                    electrode_width = 25
                else:  # Accura, Arena, Prime, Stimulo
                    electrode_width = 21

    positions = np.vstack((rows, cols)).T * mea_pitch

    probe = Probe(ndim=2, si_units="um", manufacturer="3Brain")
    probe.set_contacts(positions=positions, shapes="square", shape_params={"width": electrode_width})
    probe.annotate_contacts(row=rows)
    probe.annotate_contacts(col=cols)
    probe.create_auto_shape(probe_type="rect", margin=mea_pitch)
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def write_prb(
    file: str,
    probegroup: ProbeGroup,
    total_nb_channels: int | None = None,
    radius: float | None = None,
    group_mode: str = "by_probe",
):
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

    Parameters
    ----------
    file: str
        The name of the file to be written
    probegroup: ProbeGroup
        The Probegroup to be used for writing
    total_nb_channels: int | None, default None
        ***to do
    radius: float | None, default None
        *** to do
    group_mode: str
        One of "by_probe" or "by_shank

    """
    assert group_mode in ("by_probe", "by_shank")

    if len(probegroup.probes) == 0:
        raise ValueError("The probe group must have at least one probe")

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


def read_csv(file: str | Path):
    """
    Return a 2 or 3 columns csv file with contact positions
    """

    raise NotImplementedError


def write_csv(file, probe):
    """
    Write contact positions into a 2 or 3 columns csv file
    """

    raise NotImplementedError


_SPIKEGADGETS_NEUROPIXELS_FORMATS = {
    # SpikeConfiguration.device -> (HardwareConfiguration device name, hardcoded part number, multi-probe x-shift um)
    #
    # The SpikeGadgets .rec XML does not include a probe part number. For each
    # family (NP1 and NP2 4-shank) the listed catalogue variants share identical
    # 2D geometry in the probeinterface catalogue (contact positions, pitch,
    # stagger, shank spacing, shank width), differing only in metadata that
    # probeinterface does not consume (ADC resolution, databus phase, gain,
    # on-shank reference, shank thickness). So hardcoding one representative
    # part number produces correct geometry. `model_name` and `description` are
    # cleared on the sliced probe to avoid claiming a specific variant.
    #
    # NP1 family: NP1000, NP1001, PRB_1_2_0480_2, PRB_1_4_0480_1, PRB_1_4_0480_1_C.
    # NP2 4-shank family: NP2010, NP2013, NP2014, NP2020, NP2021.
    #
    # The multi-probe x-shift is the horizontal offset applied to successive
    # probes so they do not overlap when plotted. Chosen larger than the probe
    # width: NP1 is ~70 um wide (250 um shift leaves a generous gap); NP2
    # 4-shank is ~820 um wide (4 shanks * 250 um shank pitch + ~70 um shank
    # width), so 1000 um leaves ~180 um of gap.
    "neuropixels1": ("NeuroPixels1", "NP1000", 250.0),
    "neuropixels2": ("NeuroPixels2", "NP2014", 1000.0),
}


def read_spikegadgets(file: str | Path, raise_error: bool = True) -> ProbeGroup:
    """
    Find active channels of the given Neuropixels probe from a SpikeGadgets .rec file.
    SpikeGadgets headstages support up to three Neuropixels probes (1.0 or 2.0),
    and information for all probes will be returned in a ProbeGroup object.


    Parameters
    ----------
    file : Path or str
        The .rec file path

    Returns
    -------
    probe_group : ProbeGroup object

    """
    header_txt = parse_spikegadgets_header(file)
    root = ElementTree.fromstring(header_txt)
    hconf = root.find("HardwareConfiguration")
    sconf = root.find("SpikeConfiguration")

    # SpikeConfiguration.device selects the Neuropixels family. Default to NP1
    # when absent to preserve behavior for older files that predate the attribute.
    sconf_device = (sconf.attrib.get("device", "") if sconf is not None else "").lower()
    if sconf_device not in _SPIKEGADGETS_NEUROPIXELS_FORMATS:
        sconf_device = "neuropixels1"
    hc_device_name, part_number, multi_probe_x_shift_um = _SPIKEGADGETS_NEUROPIXELS_FORMATS[sconf_device]

    probe_configs = [d for d in hconf if d.attrib.get("name") == hc_device_name]
    n_probes = len(probe_configs)

    if n_probes == 0:
        if raise_error:
            raise Exception(f"No {hc_device_name} devices found in SpikeGadgets .rec header")
        return None

    probe_group = ProbeGroup()

    for curr_probe in range(1, n_probes + 1):
        # SpikeNTrode elements are the authoritative list of recorded electrodes.
        # Each id is "<probe_digit><1-based electrode number>"; the catalogue uses
        # 0-based electrode indices, so catalogue_index = electrode_number - 1.
        # This holds for both NP1 (up to 960 electrodes) and NP2 4-shank (up to
        # 5120 electrodes, shank-major in the catalogue: s0e0..s0e1279, s1e0..).
        #
        # The probe number is assumed to be a single digit (1, 2, or 3). This
        # matches the documented SpikeGadgets limit of three simultaneous
        # Neuropixels probes per headstage. If that limit ever changes, the
        # id-to-(probe, electrode) split will need to be revisited.
        electrode_to_hwchan = {}
        for ntrode in sconf:
            electrode_id = ntrode.attrib["id"]
            if int(electrode_id[0]) == curr_probe:
                catalogue_index = int(electrode_id[1:]) - 1
                hw_chan = int(ntrode[0].attrib["hwChan"])
                electrode_to_hwchan[catalogue_index] = hw_chan

        active_indices = np.array(sorted(electrode_to_hwchan.keys()))

        full_probe = build_neuropixels_probe(part_number)
        probe = full_probe.get_slice(active_indices)

        # Clear part-number-specific metadata since we don't know the actual part number.
        probe.model_name = ""
        probe.description = ""

        device_channels = np.array([electrode_to_hwchan[idx] for idx in active_indices])
        probe.set_device_channel_indices(device_channels)

        # Shift multiple probes so they don't overlap when plotted
        probe.move([multi_probe_x_shift_um * (curr_probe - 1), 0])

        probe_group.add_probe(probe)

    return probe_group


def parse_spikegadgets_header(file: str | Path) -> str:
    """
    Parse file (SpikeGadgets .rec format) into a string until "</Configuration>",
    which is the last tag of the header, after which the binary data begins.
    """
    header_size = None
    with open(file, mode="rb") as f:
        while True:
            line = f.readline()
            if b"</Configuration>" in line:
                header_size = f.tell()
                break

        if header_size is None:
            ValueError("SpikeGadgets: the xml header does not contain '</Configuration>'")

        f.seek(0)
        return f.read(header_size).decode("utf8")


def read_mearec(file: str | Path) -> Probe:
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

    probe = Probe(ndim=2, si_units="um", model_name=mearec_name)

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


def read_nwb(file):
    """
    Read probe position from an NWB file

    """

    raise NotImplementedError
