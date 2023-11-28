"""
This module contains useful helper functions for generating probes.

"""
from __future__ import annotations
import numpy as np

from typing import Optional

from .probe import Probe
from .probegroup import ProbeGroup
from .utils import combine_probes


_default_shape_to_params = {"circle": "radius", "square": "width", "rect": "height"}


def generate_dummy_probe(elec_shapes: "circle" | "square" | "rect" = "circle") -> Probe:
    """
    Generate a dummy probe with 3 columns and 32 contacts.
    Mainly used for testing and examples.

    Parameters
    ----------
    elec_shapes : "circle" | "square" | "rect", default: 'circle'
        Shape of the electrodes

    Returns
    -------
    probe : Probe
        The generated probe
    """

    if elec_shapes == "circle":
        contact_shape_params = {"radius": 6}
    elif elec_shapes == "square":
        contact_shape_params = {"width": 7}
    elif elec_shapes == "rect":
        contact_shape_params = {"width": 6, "height": 4.5}

    probe = generate_multi_columns_probe(
        num_columns=3,
        num_contact_per_column=[10, 12, 10],
        xpitch=25,
        ypitch=25,
        y_shift_per_column=[0, -12.5, 0],
        contact_shapes=elec_shapes,
        contact_shape_params=contact_shape_params,
    )

    probe.annotate(manufacturer="me")
    probe.annotate_contacts(quality=np.ones(32) * 1000.0)

    return probe


def generate_dummy_probe_group() -> ProbeGroup:
    """
    Generate a ProbeGroup with 2 probes.
    Mainly used for testing and examples.

    Returns
    -------
    probe : Probe
        The generated probe
    """

    probe0 = generate_dummy_probe()
    probe1 = generate_dummy_probe(elec_shapes="rect")
    probe1.move([150, -50])

    # probe group
    probegroup = ProbeGroup()
    probegroup.add_probe(probe0)
    probegroup.add_probe(probe1)

    return probegroup


def generate_tetrode(r: float = 10.0) -> Probe:
    """
    Generate a tetrode Probe.

    Parameters
    ----------
    r: float, default: 10
        The distance multiplier for the positions

    Returns
    -------
    probe : Probe
        The generated probe
    """
    probe = Probe(ndim=2, si_units="um")
    phi = np.arange(0, np.pi * 2, np.pi / 2)[:, None]
    positions = np.hstack([np.cos(phi), np.sin(phi)]) * r
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 6})
    return probe


def generate_multi_columns_probe(
    num_columns: int = 3,
    num_contact_per_column: int = 10,
    xpitch: float = 20,
    ypitch: float = 20,
    y_shift_per_column: Optional[np.array | list] = None,
    contact_shapes: "circle" | "rect" | "square" = "circle",
    contact_shape_params: dict = {"radius": 6},
) -> Probe:
    """Generate a Probe with several columns.

    Parameters
    ----------
    num_columns : int, default: 3
        Number of columns
    num_contact_per_column : int, default: 10
        Number of contacts per column
    xpitch : float, default: 20
        Pitch in x direction
    ypitch : float, default: 20
        Pitch in y direction
    y_shift_per_column : Optional[array-like], default: None
        Shift in y direction per column. It needs to have the same length as num_columns, by default None
    contact_shapes : "circle" | "rect" | "square", default: "circle"
        Shape of the contacts
    contact_shape_params : dict, default: {'radius': 6}
        Parameters for the shape.
        For circle: {"radius": float}
        For square: {"width": float}
        For rectangle: {"width": float, "height": float}

    Returns
    -------
    probe : Probe
        The generated probe
    """

    assert (
        _default_shape_to_params[contact_shapes] in contact_shape_params.keys()
    ), "contact_shapes and contact_shape_params must be coordinated see docstring"

    if isinstance(num_contact_per_column, int):
        num_contact_per_column = [num_contact_per_column] * num_columns

    if y_shift_per_column is None:
        y_shift_per_column = [0] * num_columns

    assert len(y_shift_per_column) == num_columns, (
        f"y_shift_per_column {len(y_shift_per_column)} must have " f"the same length as num_columns {num_columns}"
    )

    positions = []
    for i in range(num_columns):
        x = np.ones(num_contact_per_column[i]) * xpitch * i
        y = np.arange(num_contact_per_column[i]) * ypitch + y_shift_per_column[i]
        positions.append(np.hstack((x[:, None], y[:, None])))
    positions = np.vstack(positions)

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes=contact_shapes, shape_params=contact_shape_params)
    probe.create_auto_shape(probe_type="tip", margin=25)
    probe.set_contact_ids(np.arange(positions.shape[0]).astype("str"))

    return probe


def generate_linear_probe(
    num_elec: int = 16,
    ypitch: float = 20,
    contact_shapes: "circle" | "rect" | "square" = "circle",
    contact_shape_params: dict = {"radius": 6},
) -> Probe:
    """Generate a one-column linear probe.

    Parameters
    ----------
    num_elec : int, default: 16
        Number of electrodes
    ypitch : float, default: 20
        Pitch in y direction
    contact_shapes : "circle" | "rect" | "square", default 'circle'
        Shape of the contacts
    contact_shape_params : dict, default: {'radius': 6}
        Parameters for the shape.
        For circle: {"radius": float}
        For square: {"width": float}
        For rectangle: {"width": float, "height": float}

    Returns
    -------
    probe : Probe
        The generated probe
    """

    assert (
        _default_shape_to_params[contact_shapes] in contact_shape_params.keys()
    ), "contact_shapes and contact_shape_params must be coordinated see docstring"

    probe = generate_multi_columns_probe(
        num_columns=1,
        num_contact_per_column=num_elec,
        xpitch=0,
        ypitch=ypitch,
        contact_shapes=contact_shapes,
        contact_shape_params=contact_shape_params,
    )
    return probe


def generate_multi_shank(num_shank: int = 2, shank_pitch: list = [150, 0], **kargs) -> Probe:
    """Generate a multi-shank probe.
    Internally, calls generate_multi_columns_probe and combine_probes.

    Parameters
    ----------
    num_shank : int, default: 2
        Number of shanks
    shank_pitch : list, default: [150,0]
        Distance between shanks

    Returns
    -------
    probe : Probe
        The generated probe
    """

    shank_pitch = np.asarray(shank_pitch)

    probes = []
    for i in range(num_shank):
        probe = generate_multi_columns_probe(**kargs)
        probe.move(shank_pitch * i)
        probes.append(probe)

    multi_shank = combine_probes(probes)

    return multi_shank
