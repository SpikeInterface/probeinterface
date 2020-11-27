"""
This module give some utils function to generate probes.

"""

import numpy as np

from .probe import Probe
from .probebunch import ProbeBunch



def generate_fake_probe(elec_shapes='circle'):
    """
    Generate a 3 columns 32 channels electrode
    """
    n = 32
    positions = np.zeros((n, 2))
    for i in range(n-2):
        x = i // 10
        y = i % 10
        positions[i] = x, y
    positions *= 25
    positions[10:20, 1] -= 12.5
    positions[30] = [25, 237.5]
    positions[31] = [25, 262.5]

    probe = Probe(ndim=2, si_units='um')
    
    if elec_shapes == 'circle':
        probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 6})
    elif elec_shapes == 'square':
        probe.set_electrodes(positions=positions, shapes='square', shape_params={'width': 7})
    elif elec_shapes == 'rect':
        probe.set_electrodes(positions=positions, shapes='rect', shape_params={'width': 6, 'height': 4.5})
    
    probe.create_auto_shape(probe_type='tip', margin=25)

    return probe
    
def generate_fake_probe_bunch():
    """
    Generate a ProbeBunch with 2 probe.
    """
    probe0 = generate_fake_probe()
    probe1 = generate_fake_probe(elec_shapes='rect')
    probe1.move([150, -50])

    # probe bunch
    probebunch = ProbeBunch()
    probebunch.add_probe(probe0)
    probebunch.add_probe(probe1)
    
    return probebunch


def generate_tetrode():
    """
    Generate tetrode Probe
    
    
    """
    probe = Probe(ndim=2, si_units='um')
    phi = np.arange(0, np.pi *2, np.pi / 2)[:, None]
    positions = np.hstack([np.cos(phi), np.sin(phi)]) * 10
    probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 6})
    return probe
    

def generate_multi_columns_probe(num_columns=3, num_elec_per_column=10,
                xpitch=20, ypitch=20, y_shift_per_column=None,
                electrode_shapes='circle', electrode_shape_params={'radius': 6}):
    """
    
    
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

def generate_linear_probe(num_elec=16,  ypitch=20,
            electrode_shapes='circle', electrode_shape_params={'radius': 6}):
    probe = generate_multi_columns_probe(num_columns=1, num_elec_per_column=num_elec,
            xpitch=0, ypitch=ypitch, electrode_shapes=electrode_shapes, electrode_shape_params=electrode_shape_params)
    return probe