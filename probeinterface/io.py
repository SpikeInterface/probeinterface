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
import re
import json
from collections import OrderedDict
from packaging.version import Version, parse

import numpy as np

from .version import version
from .probe import Probe
from .probegroup import ProbeGroup


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
    with open(file, 'r', encoding='utf8') as f:
        d = json.load(f)

    # check version
    _probeinterface_format_check_version(d)

    # create probegroup from dict
    return ProbeGroup.from_dict(d)


def write_probeinterface(file, probe_or_probegroup):
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
        raise ValueError('write_probeinterface : need probe or probegroup')

    file = Path(file)

    d = OrderedDict()
    d['specification'] = 'probeinterface'
    d['version'] = version
    d.update(probegroup.to_dict(array_as_list=True))

    with open(file, 'w', encoding='utf8') as f:
        json.dump(d, f, indent=4)


tsv_label_map_to_BIDS = {'contact_ids': 'contact_id',
                         'probe_ids': 'probe_id',
                         'contact_shapes': 'contact_shape',
                         'shank_ids': 'shank_id',
                         'si_units': 'xyz_units'}
tsv_label_map_to_probeinterface = {
    v: k for k, v in tsv_label_map_to_BIDS.items()}


def read_BIDS_probe(folder, prefix=None):
    """
    Read to BIDS probe format.

    This requires a probes.tsv and a contacts.tsv file
    and potentially corresponding files in JSON format.

    Parameters
    ----------
    folder: Path or str
        The folder to scan for probes and contacts files
    prefix : None or str
        Prefix of the probes and contacts files

    Returns
    --------
    probegroup : ProbeGroup object

    """

    import pandas as pd
    folder = Path(folder)
    probes = {}
    probegroup = ProbeGroup()

    # Identify source files for probes and contacts information
    if prefix is None:
        probes_files = [f for f in folder.iterdir() if
                        f.name.endswith('probes.tsv')]
        contacts_files = [f for f in folder.iterdir() if
                          f.name.endswith('contacts.tsv')]
        if len(probes_files) != 1 or len(contacts_files) != 1:
            raise ValueError(
                'Did not find one probes.tsv and one contacts.tsv file')
        probes_file = probes_files[0]
        contacts_file = contacts_files[0]
    else:
        probes_file = folder / prefix + '_probes.tsv'
        contacts_file = folder / prefix + '_contacts.tsv'
        for file in [probes_file, contacts_file]:
            if not file.exists():
                raise ValueError(f'Source file does not exist ({file})')

    # Step 1: READING CONTACTS.TSV
    converters = {
        'x': float, 'y': float, 'z': float,
        'contact_shapes': str,
        'probe_index': int,
        'probe_id': str, 'shank_id': str, 'contact_id': str,
        'radius': float, 'width': float, 'height': float,
    }
    df = pd.read_csv(contacts_file, sep='\t', header=0,
                     keep_default_na=False, converters=converters)  #  dtype=str,
    df.replace(to_replace={'n/a': ''}, inplace=True)
    df.rename(columns=tsv_label_map_to_probeinterface, inplace=True)

    if 'probe_ids' not in df:
        raise ValueError('probes.tsv file does not contain probe_id column')
    if 'contact_ids' not in df:
        raise ValueError(
            'contacts.tsv file does not contain contact_id column')

    for probe_id in df['probe_ids'].unique():
        df_probe = df[df['probe_ids'] == probe_id].copy()

        # adding default values required by probeinterface if not present in
        # source files
        if 'contact_shapes' not in df_probe:
            df_probe['contact_shapes'] = 'circle'
            df_probe['radius'] = 1
            print(f'There is no contact shape provided for probe {probe_id}, a '
                  f'dummy circle with 1um is created')

        if 'x' not in df_probe:
            df_probe['x'] = np.arange(len(df_probe.index), dtype=float)
            print(f'There is no x coordinate provided for probe {probe_id}, a '
                  f'dummy linear x coordinate is created.')

        if 'y' not in df_probe:
            df_probe['y'] = 0.0
            print(f'There is no y coordinate provided for probe {probe_id}, a '
                  f'dummy constant y coordinate is created.')

        if 'si_units' not in df_probe:
            df_probe['si_units'] = 'um'
            print(f'There is no SI units provided for probe {probe_id}, a '
                  f'dummy SI unit (um) is created.')

        # create probe object and register with probegroup
        probe = Probe.from_dataframe(df=df_probe)
        probe.annotate(probe_id=probe_id)

        probes[str(probe_id)] = probe
        probegroup.add_probe(probe)

        ignore_annotations = ['probe_ids', 'contact_ids', 'contact_shapes', 'x',
                              'y', 'z', 'shank_ids', 'si_units',
                              'device_channel_indices', 'radius', 'width',
                              'height', 'probe_num', 'device_channel_indices']
        df_others = df_probe.drop(ignore_annotations, axis=1, errors='ignore')
        for col_name in df_others.columns:
            probe.annotate(**{col_name: df_probe[col_name].values})

    # Step 2: READING PROBES.TSV
    df = pd.read_csv(probes_file, sep='\t', header=0,
                     keep_default_na=False, dtype=str)
    df.replace(to_replace={'n/a': ''}, inplace=True)

    if 'probe_id' not in df:
        raise ValueError(
            f'{probes_file} file does not contain probe_id column')

    for row_idx, row in df.iterrows():
        probe_id = row['probe_id']
        if probe_id not in probes:
            print(f'Probe with id {probe_id} is present in probes.tsv but not '
                  f'in contacts.tsv file. Ignoring entry in probes.tsv.')
            continue

        probe = probes[probe_id]
        probe.annotate(**dict(row.items()))

        # for string based annotations use '' instead of None as default
        for string_annotation in ['name', 'manufacturer']:
            if probe.annotations.get(string_annotation, None) is None:
                probe.annotations[string_annotation] = ''

    # Step 3: READING PROBES.JSON (optional)
    probes_dict = {}
    probe_json = probes_file.with_suffix('.json')
    if probe_json.exists():
        with open(probe_json, 'r') as f:
            probes_dict = json.load(f)

    if 'ProbeId' in probes_dict:
        for probe_id, probe_info in probes_dict['ProbeId'].items():
            probe = probes[probe_id]
            for probe_param, param_value in probe_info.items():

                if probe_param == 'contour':
                    probe.probe_planar_contour = np.array(param_value)

                elif probe_param == 'units':
                    if probe.si_units is None:
                        probe.si_units = param_value
                    elif probe.si_units != param_value:
                        raise ValueError(f'Inconsistent si_units for probe '
                                         f'{probe_id}')
                else:
                    probe.annotate(**{probe_param: param_value})

    # Step 4: READING CONTACTS.JSON (optional)
    contacts_dict = {}
    contact_json = contacts_file.with_suffix('.json')
    if contact_json.exists():
        with open(contact_json, 'r') as f:
            contacts_dict = json.load(f)

    if 'ContactId' in contacts_dict:
        # collect all contact parameters used in this file
        contact_params = [k for v in contacts_dict['ContactId'].values() for k
                          in v.keys()]
        contact_params = np.unique(contact_params)

        # collect contact information for each probe_id
        for probe in probes.values():
            contact_ids = probe.contact_ids
            for contact_param in contact_params:
                # collect parameters across contact ids to add to probe
                value_list = [
                    contacts_dict['ContactId'][str(c)].get(contact_param, None)
                    for c in contact_ids]

                probe.annotate(**{contact_param: value_list})

    return probegroup


def write_BIDS_probe(folder, probe_or_probegroup, prefix=''):
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

    import pandas as pd

    if isinstance(probe_or_probegroup, Probe):
        probe = probe_or_probegroup
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
    elif isinstance(probe_or_probegroup, ProbeGroup):
        probegroup = probe_or_probegroup
    else:
        raise ValueError('probe_or_probegroup has to be'
                         'of type Probe or ProbeGroup')
    folder = Path(folder)

    # ensure that prefix and file type indicator are separated by an underscore
    if prefix != '' and prefix[-1] != '_':
        prefix = prefix + '_'

    probes = probegroup.probes

    # Step 1: GENERATION OF PROBE.TSV
    # ensure required keys (probe_id, probe_type) are present

    if any('probe_id' not in p.annotations for p in probes):
        probegroup.auto_generate_probe_ids()

    for probe in probes:
        if 'probe_id' not in probe.annotations:
            raise ValueError('Export to BIDS probe format requires '
                             'the probe id to be specified as an annotation '
                             '(probe_id). You can do this via '
                             '`probegroup.auto_generate_ids.')
        if 'type' not in probe.annotations:
            raise ValueError('Export to BIDS probe format requires '
                             'the probe type to be specified as an '
                             'annotation (type)')

    # extract all used annotation keys
    keys_by_probe = [list(p.annotations) for p in probes]
    keys_concatenated = np.concatenate(keys_by_probe)
    annotation_keys = np.unique(keys_concatenated)

    # generate a tsv table capturing probe information
    index = range(len([p.annotations['probe_id'] for p in probes]))
    df = pd.DataFrame(index=index)
    for annotation_key in annotation_keys:
        df[annotation_key] = [p.annotations[annotation_key] for p in probes]
    df['n_shanks'] = [len(np.unique(p.shank_ids)) for p in probes]

    # Note: in principle it would also be possible to add the probe width and
    # depth here based on the probe contour information. However this would
    # require an alignment of the probe within the coordinate system.

    # substitute empty values by BIDS default and create tsv file
    df.fillna('n/a', inplace=True)
    df.replace(to_replace='', value='n/a', inplace=True)
    df.to_csv(folder.joinpath(prefix + 'probes.tsv'), sep='\t', index=False)

    # Step 2: GENERATION OF PROBE.JSON
    probes_dict = {}
    for probe in probes:
        probe_id = probe.annotations['probe_id']
        probes_dict[probe_id] = {'contour': probe.probe_planar_contour.tolist(),
                                 'units': probe.si_units}
        probes_dict[probe_id].update(probe.annotations)

    with open(folder.joinpath(prefix + 'probes.json'), 'w',
              encoding='utf8') as f:
        json.dump({'ProbeId': probes_dict}, f, indent=4)

    # Step 3: GENERATION OF CONTACTS.TSV
    # ensure required contact identifiers are present
    for probe in probes:
        if probe.contact_ids is None:
            raise ValueError('Contacts must have unique contact ids '
                             'and not None for export to BIDS probe format.'
                             'Use `probegroup.auto_generate_contact_ids`.')

    df = probegroup.to_dataframe()
    index = range(sum([p.get_contact_count() for p in probes]))
    df.rename(columns=tsv_label_map_to_BIDS, inplace=True)

    df['probe_id'] = [p.annotations['probe_id'] for p in probes for _ in
                      p.contact_ids]
    df['coordinate_system'] = ['relative cartesian'] * len(index)

    channel_indices = []
    for probe in probes:
        if probe.device_channel_indices:
            channel_indices.extend(probe.device_channel_indices)
        else:
            channel_indices.extend([-1] * probe.get_contact_count())
    df['device_channel_indices'] = channel_indices

    df.fillna('n/a', inplace=True)
    df.replace(to_replace='', value='n/a', inplace=True)
    df.to_csv(folder.joinpath(prefix + 'contacts.tsv'), sep='\t', index=False)

    # Step 4: GENERATING CONTACTS.JSON
    contacts_dict = {}
    for probe in probes:
        for cidx, contact_id in enumerate(probe.contact_ids):
            cdict = {
                'contact_plane_axes': probe.contact_plane_axes[cidx].tolist()}
            contacts_dict[contact_id] = cdict

    with open(folder.joinpath(prefix + 'contacts.json'), 'w', encoding='utf8') as f:
        json.dump({'ContactId': contacts_dict}, f, indent=4)


def read_prb(file):
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
    with file.open('r') as f:
        contents = f.read()
    contents = re.sub(r'range\(([\d,]*)\)', r'list(range(\1))', contents)
    prb = {}
    exec(contents, None, prb)
    prb = {k.lower(): v for (k, v) in prb.items()}

    if 'channel_groups' not in prb:
        raise ValueError('This file is not a standard PRB file')

    probegroup = ProbeGroup()
    for i, group in prb['channel_groups'].items():
        probe = Probe(ndim=2, si_units='um')

        chans = np.array(group['channels'], dtype='int64')
        positions = np.array([group['geometry'][c] for c in chans],
                             dtype='float64')

        probe.set_contacts(positions=positions, shapes='circle',
                           shape_params={'radius': 5})
        probe.create_auto_shape(probe_type='tip')

        probe.set_device_channel_indices(chans)
        probegroup.add_probe(probe)

    return probegroup


def read_maxwell(file, well_name='well000', rec_name='rec0000'):
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

    try:
        import h5py
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)

    my_file = h5py.File(file, mode='r')

    if 'mapping' in my_file.keys():
        mapping = my_file['mapping'][:]
    else:
        mapping = my_file['wells'][well_name][rec_name]['settings']['mapping'][
            :]

    prb = {'channel_groups': {1: {}}}

    channels = list(mapping['channel'])
    electrodes = list(mapping['electrode'])
    x_pos = list(mapping['x'])
    y_pos = list(mapping['y'])
    geometry = {}
    for c, x, y in zip(channels, x_pos, y_pos):
        geometry[c] = [x, y]

    my_file.close()

    prb['channel_groups'][1]['geometry'] = geometry
    prb['channel_groups'][1]['channels'] = channels

    probe = Probe(ndim=2, si_units='um')

    chans = np.array(prb['channel_groups'][1]['channels'], dtype='int64')
    positions = np.array(
        [prb['channel_groups'][1]['geometry'][c] for c in chans],
        dtype='float64')

    probe.set_contacts(positions=positions, shapes='rect',
                       shape_params={'width': 5.45, 'height': 9.3})
    probe.annotate_contacts(electrode=electrodes)
    probe.set_planar_contour(
        ([-12.5, -12.5], [3845, -12.5], [3845, 2095], [-12.5, 2095]))

    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def read_3brain(file, mea_pitch=42, electrode_width=21):
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

    try:
        import h5py
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)

    rf = h5py.File(file, 'r')

    # get channel positions
    channels = rf['3BRecInfo/3BMeaStreams/Raw/Chs'][:]
    rows = channels['Row'] - 1
    cols = channels['Col'] - 1
    positions = np.vstack((rows, cols)).T * mea_pitch

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='square',
                       shape_params={'width': electrode_width})
    probe.annotate_contacts(row=rows)
    probe.annotate_contacts(col=cols)
    probe.create_auto_shape(probe_type='rect', margin=mea_pitch)
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def write_prb(file, probegroup,
              total_nb_channels=None,
              radius=None,
              group_mode='by_probe'
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

    """
    assert group_mode in ('by_probe', 'by_shank')

    if len(probegroup.probes) == 0:
        raise ValueError('Bad boy')

    for probe in probegroup.probes:
        if probe.device_channel_indices is None:
            raise ValueError(
                'For PRB format device_channel_indices must be set')

    with open(file, 'w') as f:
        if total_nb_channels is not None:
            f.write(f'total_nb_channels = {total_nb_channels}\n')
        if radius is not None:
            f.write(f'radius = {radius}\n')

        f.write('channel_groups = {\n')

        if group_mode == 'by_probe':
            loop = enumerate(probegroup.probes)
        elif group_mode == 'by_shank':
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


def read_spikeglx(file):
    """
    Read probe position for the meta file generated by SpikeGLX

    See http://billkarsh.github.io/SpikeGLX/#metadata-guides for implementation.

    The x_pitch/y_pitch/width are set automatically depending the NP version.

    The shape is auto generated as a shank.

    Now read:
      * NP1.0 (=phase3B2)
      * NP2.0 with 4 shank

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
    with meta_file.open(mode='r') as f:
        lines = f.read().splitlines()

    meta = {}
    for line in lines:
        if '=' not in line:
            continue
        splited = line.split('=')
        if len(splited) != 2:
            # strange lines
            continue
        k, v = splited
        if k.startswith('~'):
            # replace by the list
            k = k[1:]
            v = v[1:-1].split(')(')[1:]
        meta[k] = v

    # given this
    # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_30.md#channel-entries-by-type
    # imDatPrb_type=0/21/24
    # This is the probe type {0=NP1.0, 21=NP2.0(1-shank), 24=NP2.0(4-shank)}.
    # See also this # https://billkarsh.github.io/SpikeGLX/help/imroTables/
    # See also https://github.com/cortex-lab/neuropixels/wiki
    imDatPrb_type = int(meta.get('imDatPrb_type', 0))

    num_contact = len(meta['snsShankMap'])
    positions = np.zeros((num_contact, 2), dtype='float64')

    # the x_pitch/y_pitch depend on NP version
    if imDatPrb_type == 0:
        # NP1.0
        x_pitch = 32
        y_pitch = 20
        contact_width = 12
        shank_ids = None

        contact_ids = []
        for e in meta['imroTbl']:
            # here no elec_id is avaliable we take the chan_id instead
            chan_id = e.split(' ')[0]
            contact_ids.append(f'e{chan_id}')

        for i, e in enumerate(meta['snsShankMap']):
            x_idx = int(e.split(':')[1])
            y_idx = int(e.split(':')[2])
            stagger = np.mod(y_idx + 1, 2) * x_pitch / 2
            x_pos = x_idx * x_pitch + stagger
            y_pos = y_idx * y_pitch
            positions[i, :] = [x_pos, y_pos]

    elif imDatPrb_type == 21:
        x_pitch = 32
        y_pitch = 15
        contact_width = 12
        shank_ids = None

        contact_ids = []
        for e in meta['imroTbl']:
            elec_id = e.split(' ')[3]
            contact_ids.append(f'e{elec_id}')

        for i, e in enumerate(meta['snsShankMap']):
            x_idx = int(e.split(':')[1])
            y_idx = int(e.split(':')[2])
            x_pos = x_idx * x_pitch
            y_pos = y_idx * y_pitch
            positions[i, :] = [x_pos, y_pos]

    elif imDatPrb_type == 24:
        # NP2.0(4-shank)
        x_pitch = 32
        y_pitch = 15
        contact_width = 12
        shank_pitch = 250

        shank_ids = []
        contact_ids = []
        for e in meta['imroTbl']:
            shank_id = e.split(' ')[1]
            shank_ids.append(shank_id)
            elec_id = e.split(' ')[4]
            contact_ids.append(f's{shank_id}:e{elec_id}')

        for i, e in enumerate(meta['snsShankMap']):
            x_idx = int(e.split(':')[1])
            y_idx = int(e.split(':')[2])
            x_pos = x_idx * x_pitch + int(shank_ids[i]) * shank_pitch
            y_pos = y_idx * y_pitch
            positions[i, :] = [x_pos, y_pos]

    else:
        # NP unknown
        raise NotImplementedError(
            'This neuropixel is not implemented in probeinterface')

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='square',
                       shank_ids=shank_ids,
                       shape_params={'width': contact_width})
    probe.set_contact_ids(contact_ids)
    probe.annotate(imDatPrb_type=imDatPrb_type)

    # planar contour
    one_polygon = [(0, 10000), (0, 0), (35, -175), (70, 0), (70, 10000), ]
    if shank_ids is None:
        contour = one_polygon
    else:
        contour = []
        for i, shank_id in enumerate(np.unique(shank_ids)):
            contour += list(np.array(one_polygon) + [shank_pitch * i, 0])
    # shift
    contour = np.array(contour) - [11, 11]
    probe.set_planar_contour(contour)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe


def read_openephys(folder, settings_file=None,
                   raise_error=True):
    """
    Read probe positions from Open Ephys folder when using the Neuropix-PXI plugin.

    Parameters
    ----------
    folder : Path or str
        The path to the Open Ephys folder

    Note
    ----
    The electrode positions are only available when recording using the Neuropix-PXI plugin version >= 0.3.3

    Returns
    -------
    probe : Probe object

    """
    import xml.etree.ElementTree as ET

    folder = Path(folder).absolute()
    assert folder.is_dir()

    # find settings
    # settings_files = [p for p in folder.iterdir() if p.suffix == ".xml" and "setting" in p.name]
    settings_files = [p for p in folder.glob("**/*.xml") if "setting" in p.name]
    if len(settings_files) > 1:
        if settings_file is None:
            if raise_error:
                raise FileNotFoundError("More than one settings file found. Specify a settings "
                                           "file with the 'settings_file' argument")    
            else:
                return None
    elif len(settings_files) == 0:
        if raise_error:
            raise FileNotFoundError("Cannot find settings.xml file!")
        return None
    else:
        settings_file = settings_files[0]

    # parse xml
    tree = ET.parse(str(settings_file))
    root = tree.getroot()

    npix = None
    for child in root:
        if "SIGNALCHAIN" in child.tag:
            for proc in child:
                if "PROCESSOR" == proc.tag:
                    name = proc.attrib["name"]
                    if name == "Neuropix-PXI":
                        npix = proc
                        break
    if npix is None:
        if raise_error:
            raise Exception("Open Ephys can only be read when the Neuropix-PXI plugin is used")
        return None
    
    if parse(npix.attrib["libraryVersion"]) < parse("0.3.3"):
        if raise_error:
            raise Exception("Electrode locations are available from Neuropix-PXI version 0.3.3")
        return None

    for child in npix:
        if child.tag == "EDITOR":
            for child2 in child:
                if child2.tag == "NP_PROBE":
                    probe_name = child2.attrib["probe_name"]
                    probe_serial_number = child2.attrib["probe_serial_number"]
                    probe_part_number = child2.attrib["probe_part_number"]
                    np_probe = child2
                    break
    if np_probe is None: 
        if raise_error:
            raise Exception("Can't find 'NP_PROBE' information. Positions not available")
        return None

    # read channels
    for child in np_probe:
        if child.tag == "CHANNELS":
            channels = np.array(list(child.attrib.keys()))
    # read positions
    for child in np_probe:
        if child.tag == "ELECTRODE_XPOS":
            xpos = [float(child.attrib[ch]) for ch in channels]
        if child.tag == "ELECTRODE_YPOS":
            ypos = [float(child.attrib[ch]) for ch in channels]
    positions = np.array([xpos, ypos]).T

    # NP geometry constants
    contact_width = 12
    shank_pitch = 250

    # TODO get example of multishank for parsing
    shank_ids = None
    
    # x offset
    if "2.0" in probe_name:
        x_shift = -8
    else:
        x_shift = -11
    positions[:, 0] += x_shift

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='square',
                       shank_ids=shank_ids,
                       shape_params={'width': contact_width})
    probe.set_contact_ids(channels)
    probe.annotate(probe_name=probe_name, 
                   probe_part_number=probe_part_number,
                   probe_serial_number=probe_serial_number)

    # planar contour
    one_polygon = [(0, 10000), (0, 0), (35, -175), (70, 0), (70, 10000), ]
    if shank_ids is None:
        contour = one_polygon
    else:
        contour = []
        for i, shank_id in enumerate(np.unique(shank_ids)):
            contour += list(np.array(one_polygon) + [shank_pitch * i, 0])

    # shift
    contour = np.array(contour) - [11, 11]
    probe.set_planar_contour(contour)

    # wire it
    probe.set_device_channel_indices(np.arange(positions.shape[0]))

    return probe

    
def read_mearec(file):
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

    try:
        import h5py
    except ImportError as error:
        print(error.__class__.__name__ + ": " + error.message)

    f = h5py.File(file, "r")
    positions = f["channel_positions"][()]
    elinfo = f["info"]["electrodes"]
    elinfo_keys = elinfo.keys()

    mearec_description = None
    mearec_name = None
    if "description" in elinfo_keys:
        mearec_description = elinfo["description"][()]
    if "electrode_name" in elinfo_keys:
        mearec_name = elinfo["electrode_name"][()]

    probe = Probe(ndim=2, si_units='um')

    if "plane" in elinfo_keys:
        plane = elinfo["plane"]
    else:
        plane = "yz"  # default

    if plane == "xy":
        positions_2d = positions[()][:, :2]
    elif plane == "xz":
        positions_2d = positions[()][:, [0, 2]]
    else:
        positions_2d = positions[()][:, 1:]

    shape = None
    if "shape" in elinfo_keys:
        shape = elinfo["shape"][()]
        if isinstance(shape, bytes):
            shape = shape.decode()

    size = None
    if "shape" in elinfo_keys:
        size = elinfo["size"][()]

    shape_params = {}
    if shape is not None:
        if shape == "circle":
            shape_params = {"radius": size}
        elif shape == "square":
            shape_params = {"width": 2 * size}
        elif shape == "rect":
            shape_params = {{'width': 2 * size[0], 'height': 2 * size[1]}}

    # create contacts
    probe.set_contacts(positions_2d, shapes=shape, shape_params=shape_params)

    # add MEArec annotations
    if mearec_name is not None:
        probe.annotate(mearec_name=mearec_name)
    if mearec_description is not None:
        probe.annotate(mearec_description=mearec_description)

    # set device indices
    if elinfo["sortlist"][()] not in (b'null', 'null'):
        channel_indices = elinfo["sortlist"][()]
    else:
        channel_indices = np.arange(positions.shape[0], dtype='int64')
    probe.set_device_channel_indices(channel_indices)

    # create auto shape
    probe.create_auto_shape(probe_type='tip')

    return probe


def read_nwb(file):
    """
    Read probe position from an NWB file

    """

    raise NotImplementedError
