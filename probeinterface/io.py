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


def read_BIDS_probe(folder):
    import pandas as pd
    folder = Path(folder)
    probes = {}
    probegroup = ProbeGroup()

    if not (folder.joinpath('probes.tsv').exists() and
            folder.joinpath('contacts.tsv').exists()):
        raise ValueError(f'Could not find probes.tsv or contacts.tsv file in '
                         f'{folder}')

    # Step 1: READING PROBES.TSV
    df = pd.read_csv(folder.joinpath('probes.tsv'), sep='\t', header=0)

    if 'probe_id' not in df:
        raise ValueError('probes.tsv file does not contain probe_id column')
    for row_idx, row in df.iterrows():
        probe = Probe()
        probe.annotate(**dict(row.items()))
        probes[probe.annotations['probe_id']] = probe


    # Step 2: READING CONTACTS.TSV
    df = pd.read_csv(folder.joinpath('contacts.tsv'), sep='\t', header=0)

    if 'probe_id' not in df:
        raise ValueError('probes.tsv file does not contain probe_id column')
    if 'contact_id' not in df:
        raise ValueError('contacts.tsv file does not contain contact_id column')

    for probe_id in df['probe_id'].unique():
        df_probe = df[df['probe_id'] == probe_id]
        probe = probes[probe_id]

        probe.electrode_ids = df_probe['contact_id']
        if 'contact_shape' in df_probe:
            probe.electrode_shapes = df_probe['contact_shape']

        dimensions = []
        for dim in ['x', 'y', 'z']:
            if dim in df_probe:
                dimensions.append(dim)

        coordinates = df_probe[dimensions]
        probe.electrode_positions = coordinates
        probe.ndim = len(dimensions)

        # extract physical units used
        if 'xyz_units' in df_probe:
            if len(df['xyz_units'].unique()) == 1:
                probe.si_units = df['xyz_units'][0]
            else:
                raise ValueError(f'Probe {probe_id} uses different units for '
                                 f'contact coordinates. Can not be represented '
                                 f'by probeinterface.')

        if 'device_channel_indices' in df_probe:
            channel_indices = df_probe['device_channel_indices']
            if any([ind is not None for ind in channel_indices]):
                probe.set_device_channel_indices(channel_indices)

        used_columns = ['probe_id', 'contact_id', 'probe_shape',
                        'x', 'y', 'z', 'xyz_units']
        df_others = df_probe.drop(used_columns, axis=1, errors='ignore')
        for col_name in df_others.columns:
            probe.annotate(**dict(col_name=df_probe[col_name]))




    # TODO: continue here
    # Step 3: READING PROBES.JSON (optional)
    # Step 4: READING CONTACTS.JSON (optional)


        # to be extracted from json / position dimensions
        # shape_params = d['electrode_shape_params']
        # plane_axes = d['electrode_plane_axes'],
        # d.get('probe_planar_contour', None)
        # probe.annotate(**d['annotations'])

    for probe in probes.values():
        probegroup.add_probe(probe)

    return probegroup



def write_BIDS_probe(file, probe_or_probegroup):
    """
    Write to probe and contact formats as proposed
    for ephy BIDS extension (tsv & json based).

    The format handles several probes in one file.

    Parameters
    ----------
    file: Path or str
        The folder name.

    probe_or_probegroup : Probe or ProbeGroup
        If probe is given a probegroup is created anyway.

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
    file = Path(file)
    probes = probegroup.probes

    # Step 1: GENERATION OF PROBE.TSV
    # ensure required keys (probeID, probe_type) are present
    for probe in probes:
        # create 8-digit identifier of probe
        if 'probe_id' not in probe.annotations:
            probe.annotate(probe_id=np.random.randint(1e+7, 1e+8))
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
    df.to_csv(file.joinpath('probes.tsv'), sep='\t', index=False)


    # Step 2: GENERATION OF PROBE.JSON
    probes_dict = {}
    for probe in probes:
        probe_id = probe.annotations['probe_id']
        probes_dict[probe_id] = {'contour': probe.probe_planar_contour.tolist(),
                 'units': probe.si_units}

    with open(file.joinpath('probes.json'), 'w', encoding='utf8') as f:
        json.dump(probes_dict, f, indent=4)


    # Step 3: GENERATION OF CONTACTS.TSV
    # ensure required electrode identifiers are present
    for probe in probes:
        if probe.electrode_ids is None:
            raise ValueError('Electrodes must have unique electrode ids '
                             'and not None for export to BIDS probe format')

    index = range(sum([p.get_electrode_count() for p in probes]))
    df = pd.DataFrame(index=index)

    df['contact_id'] = [el_id for p in probes for el_id in p.electrode_ids]
    df['probe_id'] = [p.annotations['probe_id'] for p in probes for _ in p.electrode_ids]
    df['contact_shape'] = [el_shape for p in probes for el_shape in p.electrode_shapes]
    df['x'] = [x for p in probes for x in p.electrode_positions[:, 0]]
    df['y'] = [y for p in probes for y in p.electrode_positions[:, 1]]
    df['shank_id'] = [sh_id for p in probes for sh_id in p.shank_ids]
    df['coordinate_system'] = ['relative cartesian'] * len(index)
    df['xyz_units'] = [p.si_units for p in probes for _ in p.electrode_ids]

    channel_indices = []
    for probe in probes:
        if probe.device_channel_indices:
            channel_indices.extend(probe.device_channel_indices)
        else:
            channel_indices.extend([None]*probe.get_electrode_count())
    df['device_channel_indices'] = channel_indices

    df.fillna('n/a', inplace=True)
    df.replace(to_replace='', value='n/a', inplace=True)
    df.to_csv(file.joinpath('contacts.tsv'), sep='\t', index=False)


    # Step 4: GENERATING CONTACTS.JSON
    contacts_dict = {}
    for probe in probes:
        for cidx, contact_id in enumerate(probe.electrode_ids):
            cdict = {'contact_shape_params': probe.electrode_shape_params[cidx],
                     'contact_shape_units': probe.si_units}
            contacts_dict[contact_id] = cdict

    with open(file.joinpath('contacts.json'), 'w', encoding='utf8') as f:
        json.dump(contacts_dict, f, indent=4)



""" BIDS CONTACTS.TSV columns
contact_id:  REQUIRED - ID of the contact (expected to match channel.tsv)
Probe_id: REQUIRED - Id of the probe the contact is on
hemisphere:  RECOMMENDED - Which brain hemisphere was the contact located
coordinateSystem:  RECOMMENDED - coordinate system used for x,y,z
x: RECOMMENDED - recorded position along the x-axis. 
y: RECOMMENDED - recorded position along the y-axis. 
z: RECOMMENDED - recorded position along the z-axis. 
impedance:  RECOMMENDED - Impedance of the contact in kOhm.
ImpedanceUnit: RECOMMENDED - The unit of the impedance. If not specified it’s assumed to be in kOhm
Shank_id: OPTIONAL - Id to specify which shank of the probe the contact is on
contactSize: OPTIONAL - size of the contact, e.g. non-insulated surface area or length of non-insulated cable.
contact_shape: OPTIONAL - description of the shape of the contact, e.g. square, circle,
contact_size_unit: OPTIONAL - unit of the contact size. This is required if the contact size is specified.
material: OPTIONAL - material of the contact surface, e.g. Tin, Ag/AgCl, Gold
Location: RECOMMENDED - An indication on the location of the contact (e.g. cortical layer 3, CA1, etc)
Insulation: RECOMMENDED- Material used for insulation around the contact
"""








    # for probe in probegroup.probes
    # for
    #
    #     # create probe tsv
    #
    #     index = np.arange(probe.get_electrode_count(), dtype=int)
    #     df = pd.DataFrame(index=index)
    #     df['x'] = probe.electrode_positions[:, 0]
    #     df['y'] = probe.electrode_positions[:, 1]
    #     if probe.ndim == 3:
    #         df['z'] = probe.electrode_positions[:, 2]
    #     df['electrode_shapes'] = probe.electrode_shapes
    #     for i, p in enumerate(probe.electrode_shape_params):
    #         for k, v in p.items():
    #             df.at[i, k] = v
    #     df['device_channel_indices'] = probe.device_channel_indices
    #     df['shank_ids'] = probe.shank_ids
    #
    # return df

""" BIDS PROBES.TSV columns
probeID: DeviceSerialNumber (expected to match contacts.tsv)
type: REQUIRED - The type of the probe usage (acute/chronic)
CoordinateSpace: RECOMMENDED - The name of the reference space used for the coordinate definition, e.g. chamber center, or anatomical stereotactic coordinates
PlacementPicture: OPTIONAL - Path to a photograph showing the placement of the probe
x: RECOMMENDED - recorded position along the x-axis. 
y: RECOMMENDED - recorded position along the y-axis.
z: RECOMMENDED - recorded position along the z-axis.
XYZUnits: RECOMMENDED - units used for x,yz values
roll: RECOMMENDED - rotation around the roll axis
pitch: RECOMMENDED - rotation around the pitch axis
yaw: RECOMMENDED - rotation around the yaw axis
RPYUnits: RECOMMENDED - units used for roll, pitch and yaw angles
Hemisphere: RECOMMENDED - Which brain hemisphere was the probe located
Manufacturer: RECOMMENDED - Manufacturer of the probes system (e.g. "openephys”, “alphaomega",”blackrock”) 
manufacturerSerialNumber: RECOMMENDED - the serial number of the probe as provided by the manufacturer.
DeviceSerialNumber: RECOMMENDED - The serial number of the equipment (provided by the manufacturer). 
ContactCount: OPTIONAL - Number of miscellaneous analog contacts for auxiliary signals (e.g. 2). 
Depth: OPTIONAL - Physical depth of the probe, e.g. 0.3
Width: OPTIONAL Physical width of the probe, e.g. 5
Height: OPTIONAL - Physical width of the probe, e.g. 5
DimensionUnits: OPTIONAL - Units of the physical dimensions  ‘depth’, ‘width’ and ‘height’ of the probe, e.g. ‘mm’
CoordinateReferencePoint: RECOMMENDED - Point of the probe that is described by the probe coordinates
Location: RECOMMENDED - An indication on the location of the contact (e.g brain region)

"""



    # d = OrderedDict()
    # d['specification'] = 'probeinterface'
    # d['version'] = version
    # d['probes'] = []
    # for probe_ind, probe in enumerate(probegroup.probes):
    #     probe_dict = probe.to_dict(array_as_list=True)
    #     d['probes'].append(probe_dict)
    #
    # with open(file, 'w', encoding='utf8') as f:
    #     json.dump(d, f, indent=4)


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
