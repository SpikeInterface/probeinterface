"""
A simple implementation for plotting a Probe or ProbeGroup
using matplotlib.

Depending on Probe.ndim, the plotting is done in 2D or 3D
"""
from __future__ import annotations
import numpy as np
from matplotlib import path as mpl_path


def plot_probe(
    probe,
    ax=None,
    contacts_colors=None,
    with_contact_id: bool = False,
    with_device_index: bool = False,
    text_on_contact: list | np.ndarray | None = None,
    contacts_values: np.ndarray | None = None,
    cmap: str = "viridis",
    title: bool = True,
    contacts_kargs: dict = {},
    probe_shape_kwargs: dict = {},
    xlims: tuple | None = None,
    ylims: tuple | None = None,
    zlims: tuple | None = None,
    show_channel_on_click: bool = False,
):
    """Plot a Probe object.
    Generates a 2D or 3D axis, depending on Probe.ndim

    Parameters
    ----------
    probe : Probe
        The probe object
    ax : matplotlib.axis | None, default: None
        The axis to plot the probe on. If None, an axis is created
    contacts_colors : matplotlib color | None, default: None
        The color of the contacts
    with_contact_id : bool, default: False
        If True, channel ids are displayed on top of the channels
    with_device_index : bool, default: False
        If True, device channel indices are displayed on top of the channels
    text_on_contact: None | list | numpy.array, default: None
        Addintional text to plot on each contact
    contacts_values : np.array, default: None
        Values to color the contacts with
    cmap : a colormap color, default: "viridis"
         A colormap color
    title : bool, default: True
        If True, the axis title is set to the probe name
    contacts_kargs : dict, default: {}
        Dict with kwargs for contacts (e.g. alpha, edgecolor, lw)
    probe_shape_kwargs : dict, default: {}
        Dict with kwargs for probe shape (e.g. alpha, edgecolor, lw)
    xlims : tuple | None, default: None
        Limits for x dimension
    ylims : tuple | None, default: None
        Limits for y dimension
    zlims : tuple | None, default: None
        Limits for z dimension
    show_channel_on_click : bool, default: False
        If True, the channel information is shown upon click

    Returns
    -------
    poly : PolyCollection
        The polygon collection for contacts
    poly_contour : PolyCollection
        The polygon collection for the probe shape
    """
    import matplotlib.pyplot as plt

    if probe.ndim == 2:
        from matplotlib.collections import PolyCollection
    elif probe.ndim == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        if probe.ndim == 2:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.get_figure()

    _probe_shape_kwargs = dict(facecolor="green", edgecolor="k", lw=0.5, alpha=0.3)
    _probe_shape_kwargs.update(probe_shape_kwargs)

    _contacts_kargs = dict(alpha=0.7, edgecolor=[0.3, 0.3, 0.3], lw=0.5)
    _contacts_kargs.update(contacts_kargs)

    n = probe.get_contact_count()

    if contacts_colors is None and contacts_values is None:
        contacts_colors = ["orange"] * n
    elif contacts_colors is not None:
        contacts_colors = contacts_colors
    elif contacts_values is not None:
        contacts_colors = None

    vertices = probe.get_contact_vertices()
    if probe.ndim == 2:
        poly = PolyCollection(vertices, color=contacts_colors, **_contacts_kargs)
        ax.add_collection(poly)
    elif probe.ndim == 3:
        poly = Poly3DCollection(vertices, color=contacts_colors, **_contacts_kargs)
        ax.add_collection3d(poly)

    if contacts_values is not None:
        poly.set_array(contacts_values)
        poly.set_cmap(cmap)

    if show_channel_on_click:
        assert probe.ndim == 2, "show_channel_on_click works only for ndim=2"

        def on_press(event):
            return _on_press(probe, event)

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("button_release_event", on_release)

    # probe shape
    planar_contour = probe.probe_planar_contour
    if planar_contour is not None:
        if probe.ndim == 2:
            poly_contour = PolyCollection([planar_contour], **_probe_shape_kwargs)
            ax.add_collection(poly_contour)
        elif probe.ndim == 3:
            poly_contour = Poly3DCollection([planar_contour], **_probe_shape_kwargs)
            ax.add_collection3d(poly_contour)
    else:
        poly_contour = None

    if text_on_contact is not None:
        text_on_contact = np.asarray(text_on_contact)
        assert text_on_contact.size == probe.get_contact_count()

    if with_contact_id or with_device_index or text_on_contact is not None:
        if probe.ndim == 3:
            raise NotImplementedError("Channel index is 2d only")
        for i in range(n):
            txt = []
            if with_contact_id and probe.contact_ids is not None:
                contact_id = probe.contact_ids[i]
                txt.append(f"id{contact_id}")
            if with_device_index and probe.device_channel_indices is not None:
                chan_ind = probe.device_channel_indices[i]
                txt.append(f"dev{chan_ind}")
            if text_on_contact is not None:
                txt.append(f"{text_on_contact[i]}")

            txt = "\n".join(txt)
            x, y = probe.contact_positions[i]
            ax.text(x, y, txt, ha="center", va="center", clip_on=True)

    if xlims is None or ylims is None or (zlims is None and probe.ndim == 3):
        xlims, ylims, zlims = get_auto_lims(probe)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    if probe.si_units == "um":
        unit_str = "($\\mu m$)"
    else:
        unit_str = f"({probe.si_units})"
    ax.set_xlabel(f"x {unit_str}", fontsize=15)
    ax.set_ylabel(f"y {unit_str}", fontsize=15)

    if probe.ndim == 3:
        ax.set_zlim(zlims)
        ax.set_zlabel("z")

    if probe.ndim == 2:
        ax.set_aspect("equal")

    if title:
        ax.set_title(probe.get_title())

    return poly, poly_contour


def plot_probe_group(probegroup, same_axes: bool = True, **kargs):
    """Plot all probes from a ProbeGroup
    Can be in an existing set of axes or separate axes.

    Parameters
    ----------
    probegroup : ProbeGroup
        The ProbeGroup to plot
    same_axes : bool, default: True
        If True, the probes are plotted on the same axis
    kargs: dict
        see docstring for plot_probe for possible kargs
    """

    import matplotlib.pyplot as plt

    n = len(probegroup.probes)

    if same_axes:
        if "ax" in kargs:
            ax = kargs.pop("ax")
        else:
            if probegroup.ndim == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection="3d")
        axs = [ax] * n
    else:
        if "ax" in kargs:
            raise ValueError("when same_axes=False, an axes object cannot be passed into this function.")
        if probegroup.ndim == 2:
            fig, axs = plt.subplots(ncols=n, nrows=1)
            if n == 1:
                axs = [axs]
        else:
            raise NotImplementedError

    if same_axes:
        # global lims
        xlims, ylims, zlims = get_auto_lims(probegroup.probes[0])
        for i, probe in enumerate(probegroup.probes):
            xlims2, ylims2, zlims2 = get_auto_lims(probe)
            xlims = min(xlims[0], xlims2[0]), max(xlims[1], xlims2[1])
            ylims = min(ylims[0], ylims2[0]), max(ylims[1], ylims2[1])
            if zlims is not None:
                zlims = min(zlims[0], zlims2[0]), max(zlims[1], zlims2[1])
        kargs["xlims"] = xlims
        kargs["ylims"] = ylims
        kargs["zlims"] = zlims
    else:
        # will be auto for each probe in each axis
        kargs["xlims"] = None
        kargs["ylims"] = None
        kargs["zlims"] = None

    kargs["title"] = False
    for i, probe in enumerate(probegroup.probes):
        plot_probe(probe, ax=axs[i], **kargs)


def _on_press(probe, event):
    ax = event.inaxes
    x, y = event.xdata, event.ydata
    nearest_ind = np.argmin(np.sum((probe.contact_positions - np.array([[x, y]])) ** 2, axis=1))
    x_contact, y_contact = probe.contact_positions[nearest_ind, :]
    vertice = probe.get_contact_vertices()[nearest_ind]
    is_inside = mpl_path.Path(vertice).contains_points(np.array([[x, y]]))[0]
    if is_inside:
        txt = f"index {nearest_ind}"
        if probe.contact_ids is not None:
            txt += f"\n id{probe.contact_ids[nearest_ind]}"
        if probe.device_channel_indices is not None:
            txt += f"\n dev{probe.device_channel_indices[nearest_ind]}"
        t = ax.text(x_contact, y_contact, txt, color="black")
        event.canvas.draw()
        ax.contact_text = t


def on_release(event):
    ax = event.inaxes
    if hasattr(ax, "contact_text"):
        t = ax.contact_text
        t.remove()
        del ax.contact_text
        event.canvas.draw()


def get_auto_lims(probe, margin=40):
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
