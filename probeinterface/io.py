"""
Read/write some formats:
  * probeinterface h5
  * PRB (.prb)
  * CVS (.csv)
  * mearec (.h5)
  * spikeglx (.meta)
  * ironclust/jrclust (.mat)
  * NWB


"""
import os
import csv
from pathlib import Path
import re
from pprint import pformat, pprint
import json
from collections import OrderedDict

import numpy as np

from .version import version
from .probe import Probe
from .probegroup import ProbeGroup


def _probeinterface_format_check_version(d):
    """
    Check here format version for future version
    """
    pass


def read_probeinterface(file):
    """
    Read probeinterface JSON-baesd format.

    Parameters
    ----------
    
    file: Path or str
        The file name.
    
    Returns
    --------
    
    a ProbeGroup
    """
    file = Path(file)
    with open(file, 'r', encoding='utf8') as f:
        d = json.load(f)

    # check version
    _probeinterface_format_check_version(d)

    # create probegroup
    probegroup = ProbeGroup()
    for probe_dict in d['probes']:
        probe = Probe.from_dict(probe_dict)
        probegroup.add_probe(probe)
    return probegroup


def write_probeinterface(file, probe_or_probegroup):
    """
    Write to probeinterface own format JSON based.
    
    The format handles several probes in one file.
    
    Parameters
    ----------
    file: Path or str
        The file name.
    
    probe_or_probegroup : Probe or ProbeGroup
        If probe is given a probegroup is created anyway.
        
    """
    if isinstance(probe_or_probegroup, Probe):
        probe = probe_or_probegroup
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
    elif isinstance(probe_or_probegroup, ProbeGroup):
        probegroup = probe_or_probegroup
    else:
        raise ValueError('Bad boy')

    file = Path(file)

    d = OrderedDict()
    d['specification'] = 'probeinterface'
    d['version'] = version
    d['probes'] = []
    for probe_ind, probe in enumerate(probegroup.probes):
        probe_dict = probe.to_dict(array_as_list=True)
        d['probes'].append(probe_dict)

    with open(file, 'w', encoding='utf8') as f:
        json.dump(d, f, indent=4)


def read_BIDS_probe(folder, prefix=None):
    import pandas as pd
    folder = Path(folder)
    probes = {}
    probegroup = ProbeGroup()

    if prefix is None:
        probes_files = [f for f in folder.iterdir() if f.name.endswith('probes.tsv')]
        contacts_files = [f for f in folder.iterdir() if f.name.endswith('contacts.tsv')]
        if len(probes_files) != 1 or len(contacts_files) != 1:
            raise ValueError('Did not find one probes.tsv and one contacts.tsv '
                             'file')
        probe_file = probes_files[0]
        contact_file = contacts_files[0]
    else:
        raise NotImplentedError

    # Step 1: READING PROBES.TSV
    df = pd.read_csv(probe_file, sep='\t', header=0,
                     keep_default_na=False, dtype=str)
    df.replace(to_replace={'n/a': None}, inplace=True)

    if 'probe_id' not in df:
        raise ValueError('probes.tsv file does not contain probe_id column')
    for row_idx, row in df.iterrows():
        probe = Probe()
        probe.annotate(**dict(row.items()))

        # for string based annotations use '' instead of None as default
        for string_annotation in ['name', 'manufacturer']:
            if probe.annotations.get(string_annotation, None) is None:
                probe.annotations[string_annotation] = ''

    # Step 1: READING CONTACTS.TSV
    df = pd.read_csv(contacts_file, sep='\t', header=0,
                     keep_default_na=False, dtype=str)
    df.replace(to_replace={'n/a': None}, inplace=True)

    if 'probe_id' not in df:
        raise ValueError('probes.tsv file does not contain probe_id column')
    if 'contact_id' not in df:
        raise ValueError('contacts.tsv file does not contain contact_id column')

    for probe_id in df['probe_id'].unique():
        df_probe = df[df['probe_id'] == probe_id]
        probe = probes[str(probe_id)]

        probe.electrode_ids = df_probe['contact_id'].values.astype('U')
        if 'contact_shape' in df_probe:
            probe.electrode_shapes = df_probe['contact_shape'].values

        dimensions = []
        for dim in ['x', 'y', 'z']:
            if dim in df_probe:
                dimensions.append(dim)

        coordinates = df_probe[dimensions].values
        probe.electrode_positions = coordinates.astype(float)
        probe.ndim = len(dimensions)



        if 'contact_shape' in df_probe:
            shapes = df_probe['contact_shape'].values
            shape_params = 
        else:
            shapes = 'circle'
            shape_params = {'radius': 1}
            print('There is no shape contact, a dummy cricle with 1um is created')

        probe.set_electrodes(positions=dimensions, 
                    shapes=shapes, shape_params=shape_params,
                    plane_axes=None, shank_ids=None)


        # extract physical units used
        if 'xyz_units' in df_probe:
            if len(df['xyz_units'].unique()) == 1:
                probe.si_units = df['xyz_units'][0]
            else:
                raise ValueError(f'Probe {probe_id} uses different units for '
                                 f'contact coordinates. Can not be represented '
                                 f'by probeinterface.')

        if 'device_channel_indices' in df_probe:
            channel_indices = df_probe['device_channel_indices'].values
            if any([ind is not None for ind in channel_indices]):
                probe.set_device_channel_indices(channel_indices)

        if 'shank_id' in df_probe:
            shank_ids = df_probe['shank_id'].values
            probe.set_shank_ids(shank_ids)

        used_columns = ['probe_id', 'contact_id', 'contact_shape',
                        'x', 'y', 'z', 'xyz_units', 'shank_id',
                        'device_channel_indices']
        df_others = df_probe.drop(used_columns, axis=1, errors='ignore')
        for col_name in df_others.columns:
            probe.annotate(**{col_name: df_probe[col_name].values})

    # Step 3: READING PROBES.JSON (optional)
    probes_dict = {}
    probe_json = probes_file.with_suffix('.json')
    if probe_json.exists():
        with open(probe_json, 'r') as f:
            probes_dict = json.load(f)

    if 'probe_id' in probes_dict:
        for probe_id, probe_info in probes_dict['probe_id'].items():
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

    if 'contact_id' in contacts_dict:
        # collect all contact parameters used in this file
        contact_params = [k for v in contacts_dict['contact_id'].values() for k
                          in v.keys()]
        contact_params = np.unique(contact_params)

        # collect contact information for each probe_id
        for probe in probes.values():
            contact_ids = probe.contact_ids
            for contact_param in contact_params:
                # collect parameters across contact ids to add to probe
                value_list = [
                    contacts_dict['contact_id'][str(c)].get(contact_param, None)
                    for c in contact_ids]

                if contact_param == 'contact_shape_params':
                    probe.electrode_shape_params = value_list
                elif contact_param == 'contact_plane_axes':
                    probe.electrode_plane_axes = np.array(value_list)

                # extract units if this was not extracted earlier already
                elif contact_param == 'contact_shape_units' and \
                        all(x == value_list[0] for x in value_list):
                    if probe.si_units == None:
                        probe.si_units = value_list[0]
                    elif probe.si_units != value_list[0]:
                        raise ValueError(f'Different units used for probe and '
                                         f'contacts ({probe.si_units} != '
                                         f'{value_list[0]})')
                else:
                    probe.annotate(**{contact_param: value_list})

    for probe in probes.values():
        probegroup.add_probe(probe)

    return probegroup


def write_BIDS_probe(folder, probe_or_probegroup, prefix=''):
    """
    Write to probe and contact formats as proposed
    for ephy BIDS extension (tsv & json based).

    The format handles several probes in one file.

    Parameters
    ----------
    folder: Path or str
        The folder name.

    probe_or_probegroup : Probe or ProbeGroup
        If probe is given a probegroup is created anyway.

    prefix: str
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
    # ensure required keys (probeID, probe_type) are present

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

    with open(folder.joinpath(prefix + 'probes.json'), 'w',
              encoding='utf8') as f:
        json.dump({'probe_id': probes_dict}, f, indent=4)

    # Step 3: GENERATION OF CONTACTS.TSV
    # ensure required contact identifiers are present
    for probe in probes:
        if probe.contact_ids is None:
            raise ValueError('Contacts must have unique contact ids '
                             'and not None for export to BIDS probe format.'
                             'Use `probegroup.auto_generate_contact_ids`.')

    df = probegroup.to_dataframe()
    index = range(sum([p.get_contact_count() for p in probes]))
    df.rename(columns=label_map_to_BIDS, inplace=True)

    df['probe_id'] = [p.annotations['probe_id'] for p in probes for _ in
                      p.contact_ids]
    df['coordinate_system'] = ['relative cartesian'] * len(index)

    channel_indices = []
    for probe in probes:
        if probe.device_channel_indices:
            channel_indices.extend(probe.device_channel_indices)
        else:
            channel_indices.extend([None] * probe.get_contact_count())
    df['device_channel_indices'] = channel_indices

    df.fillna('n/a', inplace=True)
    df.replace(to_replace='', value='n/a', inplace=True)
    df.to_csv(folder.joinpath(prefix + 'contacts.tsv'), sep='\t', index=False)

    # Step 4: GENERATING CONTACTS.JSON
    contacts_dict = {}
    for probe in probes:
        for cidx, contact_id in enumerate(probe.contact_ids):
            cdict = {'contact_plane_axes': probe.contact_plane_axes[cidx].tolist()}
            contacts_dict[contact_id] = cdict

    with open(folder.joinpath(prefix + 'contacts.json'), 'w', encoding='utf8') as f:
        json.dump({'contact_id': contacts_dict}, f, indent=4)


def read_prb(file):
    """
    Read a PRB file and return a ProbeGroup object.
    
    Since PRB do not handle contact shape then circle of 5um are put.
    Same for contact shape a dummy tip is put.
    
    PRB format do not contain any information about the channel of the probe
    Only the channel index on device is given.
    
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
    Read a maxwell file and return a Probe object. Maxwell file format can be
    either Maxone (and thus just the file name is needed), or MaxTwo. In case
    of the latter, you need to explicitly specify what is the well number of 
    interest (well000 by default), and the recording session (since there can
    be several. Default is rec0000)

    Since Maxwell do not handle contact shape then circle of 5um are put.
    Same for contact shape a dummy tip is put.

    Maxwell format do not contain any information about the channel of the probe
    Only the channel index on device is given. 

    Parameters
    ----------
    
    file: Path or str
        The file name.

    well_name: str
        If MaxTwo file format, the well_name to extract the mapping from
        (default is well000)

    rec_name: str
        If MaxTwo file format, the recording session to extract the mapping
        from (default is rec0000)
    
    Returns
    --------
    
    a Probe

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
    probe.set_planar_contour(
        ([-12.5, -12.5], [3845, -12.5], [3845, 2095], [-12.5, 2095]))

    probe.set_device_channel_indices(chans)

    return probe


def write_prb(file, probegroup):
    """
    Write ProbeGroup into a prb file.
    
    This format handle:
      * multi Probe with channel group index key
      * channel positions with "geometry"
      * device_channel_indices with "channels "key
    
    Note: many information are lost in the PRB format:
      * contact shape
      * shape
      * channel index
    
    Note:
      * "total_nb_channels" is not handle because it is a non sens key
      * "graph" is not handle because it is useless
      * "radius" is not handle. It was only for early version of spyking-cicus
    """
    if len(probegroup.probes) == 0:
        raise ValueError('Bad boy')

    for probe in probegroup.probes:
        if probe.device_channel_indices is None:
            raise ValueError(
                'For PRB format device_channel_indices must be set')

    with open(file, 'w') as f:
        f.write('channel_groups = {\n')

        for probe_ind, probe in enumerate(probegroup.probes):
            f.write(f"    {probe_ind}:\n")
            f.write("        {\n")
            channels = probe.device_channel_indices
            keep = channels >= 0
            #  channels -1 are not wired

            chans = list(channels[keep])
            f.write(f"           'channels': {chans},\n")
            f.write("           'geometry':  {\n")
            for c in range(probe.get_contact_count()):
                if not keep[c]:
                    continue
                pos = list(probe.contact_positions[c, :])
                f.write(f"               {channels[c]}: {pos},\n")
            f.write("           }\n")
            f.write("       },\n")

        f.write("}\n")


def read_cvs(file):
    """
    Return a 2 or 3 columns csv file with contacts position
    """
    raise NotImplementedError


def write_cvs(file, probe):
    """
    Write probe postions into a 2 or 3 columns csv file
    """
    raise NotImplementedError


def read_spikeglx(file):
    """
    read probe position for the meta file generated by spikeglx
    
    See http://billkarsh.github.io/SpikeGLX/#metadata-guides for implementation.
    
    The x_pitch/y_pitch/width are settle automatically depending the NP version
    
    The shape is a dummy one for the moment.
    
    Now read NP1.0 (=phase3B2)
    
    Return Probe object
    """
    meta_file = Path(file)
    assert meta_file.suffix == ".meta", "'meta_file' should point to the .meta SpikeGLX file"
    with meta_file.open(mode='r') as f:
        lines = f.read().splitlines()

    meta = {}
    for line in lines:
        k, v = line.split('=')
        if k.startswith('~'):
            # replace by the list
            k = k[1:]
            v = v[1:-1].split(')(')[1:]
        meta[k] = v
    #~ from pprint import pprint
    #~ pprint(meta)
    
    # given this
    # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_30.md#channel-entries-by-type
    # imDatPrb_type=0/21/24
    # This is the probe type {0=NP1.0, 21=NP2.0(1-shank), 24=NP2.0(4-shank)}.
    
    # older file don't have this field
    imDatPrb_type = int(meta.get('imDatPrb_type', 0))
    
    # the x_pitch/y_pitch depend on NP version
    if imDatPrb_type == 0:
        # NP1.0
        x_pitch=32
        y_pitch=20
        width=12
    elif imDatPrb_type == 21:
        #21=NP2.0(1-shank)
        raise NotImplementedError('NP2.0(1-shank) is not implemenetd yet')
    elif imDatPrb_type == 24:
        # NP2.0(4-shank)
        raise NotImplementedError('NP2.0(4-shank) is not implemenetd yet')
    else:
        #NP unknown
        raise NotImplementedError
    
    
    positions = []
    for e in meta['snsShankMap']:
        x_idx = int(e.split(':')[1])
        y_idx = int(e.split(':')[2])
        stagger = np.mod(y_idx + 1, 2) * x_pitch / 2
        x_pos = x_idx * x_pitch + stagger
        y_pos = y_idx * y_pitch
        positions.append([x_pos, y_pos])
    positions = np.array(positions)

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='square',
                         shape_params={'width': 10})
    probe.create_auto_shape(probe_type='tip')

    return probe


def read_mearec(file):
    """
    read probe position, and contact shape from a mearec file
    
    Alesio : this is for you
    """
    raise NotImplementedError


def read_nwb(file):
    """
    read probe position from the NWB format
    """
    raise NotImplementedError


# OLD hdf5 implementation
'''

def read_probeinterface(file):
    """
    Read probeinterface own format hdf5 based.
    
    Implementation is naive but ot works.
    """
    import h5py

    probegroup = ProbeGroup()

    file = Path(file)
    with h5py.File(file, 'r') as f:
        for key in f.keys():
            if key.startswith('probe_'):
                probe_ind = int(key.split('_')[1])
                probe_dict = {}
                for k in Probe._dump_attr_names:
                    path = f'/{key}/{k}'
                    if not path in f:
                        continue
                    v = f[path]
                    if k == 'contact_shapes':
                        v2 = np.array(v).astype('U')
                    elif k == 'contact_shape_params':
                        l = []
                        for e in v:
                            d = {}
                            exec(e.decode(), None, d)
                            l.append(d['value'])
                        v2 = np.array(l, dtype='O')
                    elif k == 'si_units':
                        v2 = str(v[0])
                    elif k == 'ndim':
                        v2 = int(v[0])
                    else:
                        v2 = np.array(v)
                    probe_dict[k] = v2

                probe = Probe.from_dict(probe_dict)
                probegroup.add_probe(probe)
    return probegroup


def write_probeinterface(file, probe_or_probegroup):
    """
    Write to probeinterface own format hdf5 based.
    
    Implementation is naive but ot works.
    """
    import h5py
    if isinstance(probe_or_probegroup, Probe):
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
    elif isinstance(probe_or_probegroup, ProbeGroup):
        probegroup = probe_or_probegroup
    else:
        raise valueError('Bad boy')

    file = Path(file)

    with h5py.File(file, 'w') as f:
        for probe_ind, probe in enumerate(probegroup.probes):
            d = probe.to_dict()
            for k, v in d.items():
                if k == 'contact_shapes':
                    v2 = v.astype('S')
                elif k == 'contact_shape_params':
                    v2 = np.array(['value=' + pformat(e) for e in v], dtype='S')
                elif k == 'si_units':
                    v2 = np.array([v.encode('utf8')])
                elif k == 'ndim':
                    v2 = np.array([v])
                else:
                    v2 = v
                path = f'/probe_{probe_ind}/{k}'
                f.create_dataset(path, data=v2)
'''
