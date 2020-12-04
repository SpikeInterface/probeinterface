"""
Some utility functions
"""
import numpy as np

from .probe import Probe

def combinate_probes(probes):
    """
    Combinate several Probe object into a unique
    Probe object multi multi shank.
    
    This work only for ndim=2

    This will have strange behavrior if:
      * probes have been rotated
      * probes NOT have been moved (probe overlap in space )
    
    
    """

    # check ndim
    assert all(probes[0].ndim == p.ndim for p in probes)
    assert probes[0].ndim ==2


    n = sum(p.get_electrode_count() for p in probes)
    print('n', n)


    
    kwargs = {}
    for k in ('electrode_positions', 'electrode_plane_axes',
                        'electrode_shapes', 'electrode_shape_params'):
        v = np.concatenate([getattr(p, k) for p in probes], axis=0)
        kwargs[k.replace('electrode_', '')] = v

    shank_ids = np.concatenate([np.ones(p.get_electrode_count(), dtype='int64') * i
                                                     for i,p in enumerate(probes)])
    kwargs['shank_ids'] = shank_ids
    
    # TODO deal with electrode_ids/device_channel_indices

    multi_shank = Probe(ndim=probes[0].ndim, si_units=probes[0].si_units)
    multi_shank.set_electrodes(**kwargs)

    # global shape
    have_shape = all(p.probe_shape_vertices is not None for p in probes)
    
    if have_shape:
        verts = np.concatenate([p.probe_shape_vertices for p in probes], axis=0)
        multi_shank.set_shape_vertices(verts)

    return multi_shank