"""
Some utility functions
"""
import numpy as np

from .probe import Probe


def combine_probes(probes, connect_shape=True):
    """
    Combine several Probe objects into a unique
    multi-shank Probe object

    This works only when ndim=2

    This will have strange behavior if:
      * probes have been rotated
      * one of the probes has NOT been moved from its original location
       (results in probes overlapping in space )


    Parameters
    ----------
    probes : list of Probe

    connect_shape : bool (default True)
        Connect all shapes togother.
        Be careful, as this can lead to strange probe shape....

    Return
    ----------
    multi_shank : a multi-shank Probe object

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

def generate_unique_ids(min, max, n, trials=20):
    """
    Create n unique identifiers

    Creates `n` unique integer identifiers between `min` and `max` within a
    maximum number of `trials` attempts

    Parameters
    ----------
    min (int) : minimal value permitted for an identifier
    max (int) : maximal value permitted for an identifier
    n (int) : number of identifiers to create
    trials (int): maximal number of attempts for generating the set of
        identifiers

    Returns
    -------
    A numpy array of `n` unique integer identifiers

    """

    ids = np.random.randint(min, max, n)
    i = 0

    while len(np.unique(ids)) != len(ids) and i < trials:
        ids = np.random.randint(min, max, n)

    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'Can not generate {n} unique ids between {min} '
                         f'and {max} in {trials} trials')
    return ids
