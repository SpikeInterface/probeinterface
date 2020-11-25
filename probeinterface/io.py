"""
Read/write some formats:
  * PRB
  * 

"""
import csv
from pathlib import Path

import numpy as np

from .probe import Probe
from .probebunch import ProbeBunch


def read_python(path):
    '''Parses python scripts in a dictionary

    Parameters
    ----------
    path: str or Path
        Path to file to parse

    Returns
    -------
    metadata:
        dictionary containing parsed file

    '''
    from six import exec_
    import re
    path = Path(path).absolute()
    assert path.is_file()
    with path.open('r') as f:
        contents = f.read()
    contents = re.sub(r'range\(([\d,]*)\)',r'list(range(\1))',contents)
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def read_prb(file):
    """
    Read a PRB file and return a ProbeBunch object.
    
    Since PRB do not handle electrode shape then circle of 5um are put.
    Same for electrode shape a fake tip is put.
    
    PRB format do not contain any information about the channel of the probe
    Only the channel index on device is given.
    
    """
    prb = read_python(file)
    if 'channel_groups' not in prb:
        raise ValueError('This file is not a standard PRB file')
    
    probebunch = ProbeBunch()
    for i, group in prb['channel_groups'].items():
        probe = Probe(ndim=2, si_units='um')
        
        chans = np.array(group['channels'], dtype='int64')
        positions = np.array([group['geometry'][c] for c in chans], dtype='float64')
        
        probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 5})
        probe.create_auto_shape(probe_type='tip')
        
        
        probe.set_device_channel_indices(chans)
        probebunch.add_probe(probe)
    
    return probebunch

def write_prb(file, probebunch):
    """
    Write ProbeBunch into a prb file.
    
    This format handle:
      * multi Probe with channel group index key
      * channel positions with "geometry"
      * device_channel_indices with "channels "key
    
    Note: many information are lost in the PRB format:
      * electrode shape
      * shape
      * channel index
    """
    raise NotImplementedError
    

    
    