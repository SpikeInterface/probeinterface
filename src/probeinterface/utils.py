"""
Some utility functions
"""

from __future__ import annotations
from importlib import import_module
from types import ModuleType

import numpy as np

from .probe import Probe


def import_safely(module: str) -> ModuleType:
    """
    Safely import a module with importlib and return the imported module object.

    Parameters
    ----------
    module : str
        The name of the module to import.

    Returns
    -------
    module_obj : module
        The imported module object.

    Raises
    ------
    ImportError
        If the specified module cannot be imported.

    Examples
    --------
    >>> import math
    >>> math_module = import_safely("math")
    >>> math_module.sqrt(4)
    2.0

    >>> import_safely("non_existent_module")
    ImportError: No module named 'non_existent_module'
    """

    try:
        module_obj = import_module(module)
    except ImportError as error:
        raise ImportError(f"{repr(error)}")

    return module_obj


def combine_probes(probes: list[Probe], connect_shape: bool = True) -> Probe:
    """
    Combine several Probe objects into a unique
    multi-shank Probe object.
    This works only when ndim=2

    This will have strange behavior if:
      * probes have been rotated
      * one of the probes has NOT been moved from its original location
        (results in probes overlapping in space )


    Parameters
    ----------
    probes : list
        List of Probe objects
    connect_shape : bool, default: True
        Connect all shapes together.
        Be careful, as this can lead to strange probe shape....

    Return
    ----------
    multi_shank : a multi-shank Probe object

    """

    # check ndim
    assert all(probes[0].ndim == p.ndim for p in probes), "all probes must have the same ndim"
    assert probes[0].ndim == 2, "All probes should be 2d"

    kwargs = {}
    for k in ("contact_positions", "contact_plane_axes", "contact_shapes", "contact_shape_params"):
        v = np.concatenate([getattr(p, k) for p in probes], axis=0)
        kwargs[k.replace("contact_", "")] = v

    shank_ids = np.concatenate([np.ones(p.get_contact_count(), dtype="int64") * i for i, p in enumerate(probes)])
    kwargs["shank_ids"] = shank_ids

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


def generate_unique_ids(min: int, max: int, n: int, trials: int = 20) -> np.array:
    """
    Create n unique identifiers.
    Creates `n` unique integer identifiers between `min` and `max` within a
    maximum number of `trials` attempts.

    Parameters
    ----------
    min : int
        Minimun value permitted for an identifier
    max : int
        Maximum value permitted for an identifier
    n : int
        Number of identifiers to create
    trials : int, default: 20
        Maximal number of attempts for generating
        the set of identifiers

    Returns
    -------
    ids : A numpy array of `n` unique integer identifiers

    """

    ids = np.random.randint(min, max, n)
    i = 0

    while len(np.unique(ids)) != len(ids) and i < trials:
        ids = np.random.randint(min, max, n)

    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Can not generate {n} unique ids between {min} " f"and {max} in {trials} trials")
    return ids


def get_auto_lims(probe: Probe, margin: float = 40.0) -> tuple[float, float, float]:
    """
    Compute the boundaries of a given probe, considering its contour and an optional margin.
    The function is designed to handle both planar and three-dimensional probes.

    Parameters
    ----------
    probe : Probe
        The probe for which the limits are to be computed.
    margin : float, default: 40
        An isotropic margin that is added to the exact probe boundaries.

    Returns
    -------
    lims : a tuple containing the limits in the x, y, and z directions
           (xlims, ylims, zlims). If the provided probe is planar, then
           zlims is None.
    """
    positions = probe.contact_positions
    planar_contour = probe.probe_planar_contour

    xlims = np.min(positions[:, 0]), np.max(positions[:, 0])
    ylims = np.min(positions[:, 1]), np.max(positions[:, 1])
    zlims = None

    if probe.ndim == 3:
        zlims = np.min(positions[:, 2]), np.max(positions[:, 2])

    if planar_contour is not None:
        xlims2 = np.min(planar_contour[:, 0]), np.max(planar_contour[:, 0])
        xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])

        ylims2 = np.min(planar_contour[:, 1]), np.max(planar_contour[:, 1])
        ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])

        if probe.ndim == 3:
            zlims2 = np.min(planar_contour[:, 2]), np.max(planar_contour[:, 2])
            zlims = min(zlims[0], zlims2[0]), max(zlims[1], zlims2[1])

    xlims = xlims[0] - margin, xlims[1] + margin
    ylims = ylims[0] - margin, ylims[1] + margin

    if probe.ndim == 3:
        zlims = zlims[0] - margin, zlims[1] + margin

        # to keep equal aspect in 3d
        # all axes have the same limits
        lims = min(xlims[0], ylims[0], zlims[0]), max(xlims[1], ylims[1], zlims[1])
        xlims, ylims, zlims = lims, lims, lims

    return xlims, ylims, zlims
