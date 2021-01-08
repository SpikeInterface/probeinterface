"""
This module give some utils function to generate probes.

"""

import numpy as np

from .probe import Probe
from .probegroup import ProbeGroup
from .utils import combine_probes


def generate_dummy_probe(elec_shapes='circle'):
    """
    Generate a 3 columns 32 channels electrode.
    Mainly used for testing and examples.
    """
    if elec_shapes == 'circle':
        electrode_shape_params = {'radius': 6}
    elif elec_shapes == 'square':
        electrode_shape_params = {'width': 7}
    elif elec_shapes == 'rect':
        electrode_shape_params = {'width': 6, 'height': 4.5}

    probe = generate_multi_columns_probe(num_columns=3,
                                         num_elec_per_column=[10, 12, 10],
                                         xpitch=25, ypitch=25, y_shift_per_column=[0, -12.5, 0],
                                         electrode_shapes=elec_shapes, electrode_shape_params=electrode_shape_params)

    return probe


def generate_dummy_probe_group():
    """
    Generate a ProbeGroup with 2 probe.
    Mainly used for testing and examples.
    """
    probe0 = generate_dummy_probe()
    probe1 = generate_dummy_probe(elec_shapes='rect')
    probe1.move([150, -50])

    # probe group
    probegroup = ProbeGroup()
    probegroup.add_probe(probe0)
    probegroup.add_probe(probe1)

    return probegroup


def generate_tetrode(r=10):
    """
    Generate tetrode Probe
    
    
    """
    probe = Probe(ndim=2, si_units='um')
    phi = np.arange(0, np.pi * 2, np.pi / 2)[:, None]
    positions = np.hstack([np.cos(phi), np.sin(phi)]) * r
    probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 6})
    return probe


def generate_multi_columns_probe(num_columns=3, num_elec_per_column=10,
                                 xpitch=20, ypitch=20, y_shift_per_column=None,
                                 electrode_shapes='circle', electrode_shape_params={'radius': 6}):
    """
    Generate a Probe with several columns
    """

    if isinstance(num_elec_per_column, int):
        num_elec_per_column = [num_elec_per_column] * num_columns

    if y_shift_per_column is None:
        y_shift_per_column = [0] * num_columns

    positions = []
    for i in range(num_columns):
        x = np.ones(num_elec_per_column[i]) * xpitch * i
        y = np.arange(num_elec_per_column[i]) * ypitch + y_shift_per_column[i]
        positions.append(np.hstack((x[:, None], y[:, None])))
    positions = np.vstack(positions)

    probe = Probe(ndim=2, si_units='um')
    probe.set_electrodes(positions=positions, shapes=electrode_shapes,
                         shape_params=electrode_shape_params)
    probe.create_auto_shape(probe_type='tip', margin=25)

    return probe


def generate_linear_probe(num_elec=16, ypitch=20,
                          electrode_shapes='circle', electrode_shape_params={'radius': 6}):
    """
    Generate a linear Probe (one columns)
    """

    probe = generate_multi_columns_probe(num_columns=1, num_elec_per_column=num_elec,
                                         xpitch=0, ypitch=ypitch, electrode_shapes=electrode_shapes,
                                         electrode_shape_params=electrode_shape_params)
    return probe


def generate_multi_shank(num_shank=2, shank_pitch=[150, 0], **kargs):
    """
    Generate a multi shank probe.
    Internally do a call to generate_multi_columns_probe
    and use combine_probes.
    """
    shank_pitch = np.asarray(shank_pitch)

    probes = []
    for i in range(num_shank):
        probe = generate_multi_columns_probe(**kargs)
        probe.move(shank_pitch * i)
        probes.append(probe)

    multi_shank = combine_probes(probes)

    return multi_shank
