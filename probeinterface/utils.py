"""
Some utility functions
"""
import numpy as np

from .probe import Probe


def combine_probes(probes, connect_shape=True):
    """
    Combinate several Probe object into a unique
    Probe object multi multi shank.
    
    This work only for ndim=2

    This will have strange behavrior if:
      * probes have been rotated
      * probes NOT have been moved (probe overlap in space )
    
    
    Parameters
    ----------
    probes: list of Probe
    
    connect_shape: bool (default True)
        Connect all shape togother.
        This can lead to strange probe shape....

    Return
    ----------
    A multi-shank probe object.
    
    """

    # check ndim
    assert all(probes[0].ndim == p.ndim for p in probes)
    assert probes[0].ndim == 2

    n = sum(p.get_contact_count() for p in probes)

    kwargs = {}
    for k in ('contact_positions', 'contact_plane_axes',
              'contact_shapes', 'contact_shape_params'):
        v = np.concatenate([getattr(p, k) for p in probes], axis=0)
        kwargs[k.replace('contact_', '')] = v

    shank_ids = np.concatenate([np.ones(p.get_contact_count(), dtype='int64') * i
                                for i, p in enumerate(probes)])
    kwargs['shank_ids'] = shank_ids

    # TODO deal with contact_ids/device_channel_indices

    multi_shank = Probe(ndim=probes[0].ndim, si_units=probes[0].si_units)
    multi_shank.set_contacts(**kwargs)

    # global shape
    have_shape = all(p.probe_planar_contour is not None for p in probes)

    if have_shape and connect_shape:
        verts = np.concatenate([p.probe_planar_contour for p in probes], axis=0)
        verts = np.concatenate([verts[0:1] + [0, 40], verts, verts[-1:] + [0, 40]], axis=0)

        multi_shank.set_planar_contour(verts)

    return multi_shank
