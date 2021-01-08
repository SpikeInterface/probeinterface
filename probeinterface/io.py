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
    
    The format handle several probes in one file.
    
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
        raise valueError('Bad boy')

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


def read_prb(file):
    """
    Read a PRB file and return a ProbeGroup object.
    
    Since PRB do not handle electrode shape then circle of 5um are put.
    Same for electrode shape a dummy tip is put.
    
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
        positions = np.array([group['geometry'][c] for c in chans], dtype='float64')

        probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 5})
        probe.create_auto_shape(probe_type='tip')

        probe.set_device_channel_indices(chans)
        probegroup.add_probe(probe)

    return probegroup


def write_prb(file, probegroup):
    """
    Write ProbeGroup into a prb file.
    
    This format handle:
      * multi Probe with channel group index key
      * channel positions with "geometry"
      * device_channel_indices with "channels "key
    
    Note: many information are lost in the PRB format:
      * electrode shape
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
            raise ValueError('For PRB format device_channel_indices must be set')

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
            for c in range(probe.get_electrode_count()):
                if not keep[c]:
                    continue
                pos = list(probe.electrode_positions[c, :])
                f.write(f"               {channels[c]}: {pos},\n")
            f.write("           }\n")
            f.write("       },\n")

        f.write("}\n")


def read_cvs(file):
    """
    Return a 2 or 3 columns csv file with electrodes position
    """
    raise NotImplementedError


def write_cvs(file, probe):
    """
    Write probe postions into a 2 or 3 columns csv file
    """
    raise NotImplementedError


def read_spikeglx(file, ):
    """
    read probe position for the meta file generated by spikeglx
    """

    with open(file, mode='r') as f:
        lines = f.read().splitlines()

    meta = {}
    for line in lines:
        k, v = line.split('=')
        if k.startswith('~'):
            # replace by the list
            k = k[1:]
            v = v[1:-1].split(')(')[1:]
        meta[k] = v

    #  TODO make pitch and width more accurate depending the NP version

    # Here for NP 1.0
    x_pitch = 16
    y_pitch = 20
    width = 12

    positions = []
    for e in meta['snsShankMap']:
        x_pos = int(e.split(':')[1])
        y_pos = int(e.split(':')[2])
        positions.append([x_pos, y_pos])

    positions = np.array(positions)
    positions[:, 0] *= x_pitch
    positions[:, 1] *= y_pitch

    probe = Probe(ndim=2, si_units='um')
    probe.set_electrodes(positions=positions, shapes='square', shape_params={'width': 10})
    probe.create_auto_shape(probe_type='tip')

    return probe


def read_mearec(file):
    """
    read probe position, and electrode shape from a mearec file
    
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
                    if k == 'electrode_shapes':
                        v2 = np.array(v).astype('U')
                    elif k == 'electrode_shape_params':
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
                if k == 'electrode_shapes':
                    v2 = v.astype('S')
                elif k == 'electrode_shape_params':
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
