"""
A simple implementation for plotting a Probe or ProbeGroup
using matplotlib.

Depending on Probe.ndim, the plotting is done in 2D or 3D
"""

import numpy as np


def plot_probe(probe, ax=None, contacts_colors=None,
                with_channel_index=False, with_contact_id=False, 
                with_device_index=False,
                first_index='auto',
                contacts_values=None, cmap='viridis',
                title=True, contacts_kargs={}, probe_shape_kwargs={},
                xlims=None, ylims=None, zlims=None):
    """
    plot one probe.
    switch to 2D or 3D, depending on Probe.ndim

    """

    import matplotlib.pyplot as plt
    if probe.ndim == 2:
        from matplotlib.collections import PolyCollection
    elif probe.ndim == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        if probe.ndim == 2:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')

    if first_index == 'auto':
        if 'first_index' in probe.annotations:
            first_index = probe.annotations['first_index']
        elif probe.annotations.get('manufacturer', None) == 'neuronexus':
            # neuronexus is one based indexing
            first_index = 1
        else:
            first_index = 0
    assert first_index in (0, 1)

    _probe_shape_kwargs = dict(facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
    _probe_shape_kwargs.update(probe_shape_kwargs)

    _contacts_kargs = dict(alpha=0.7, edgecolor=[0.3, 0.3, 0.3], lw=0.5)
    _contacts_kargs.update(contacts_kargs)

    n = probe.get_contact_count()

    if contacts_colors is None and contacts_values is None:
        contacts_colors = ['orange'] * n
    elif contacts_colors is not None:
        contacts_colors = contacts_colors
    elif contacts_values is not None:
        contacts_colors = None

    # contacts
    positions = probe.contact_positions

    vertices = probe.get_contact_vertices()
    if probe.ndim == 2:
        poly = PolyCollection(vertices, color=contacts_colors, **_contacts_kargs)
        ax.add_collection(poly)
    elif probe.ndim == 3:
        poly =  Poly3DCollection(vertices, color=contacts_colors, **_contacts_kargs)
        ax.add_collection3d(poly)

    if contacts_values is not None:
        poly.set_array(contacts_values)
        poly.set_cmap(cmap)


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


    if with_channel_index or with_contact_id or  with_device_index:
        if probe.ndim == 3:
            raise NotImplementedError('Channel index is 2d only')
        for i in range(n):
            txt = []
            if with_channel_index:
                txt.append(f'{i + first_index}')
            if with_contact_id and probe.contact_ids is not None:
                contact_id = probe.contact_ids[i]
                txt.append(f'id{contact_id}')
            if with_device_index and probe.device_channel_indices is not None:
                chan_ind = probe.device_channel_indices[i]
                txt.append(f'dev{chan_ind}')
            # txt = ':'.join(txt)
            txt = '\n'.join(txt)
            x, y = probe.contact_positions[i]
            ax.text(x, y, txt, ha='center', va='center')

    if xlims is None or ylims is None or (zlims is None and probe.ndim == 3):
        xlims, ylims, zlims = get_auto_lims(probe)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if probe.ndim == 3:
        ax.set_zlim(zlims)
        ax.set_zlabel('z')

    if probe.ndim == 2:
        ax.set_aspect('equal')

    if title:
        ax.set_title(probe.get_title())

    return poly, poly_contour

def plot_probe_group(probegroup, same_axes=True, **kargs):
    """
    Plot all probes from a ProbeGroup

    Can be in an existing set of axes or separate axes.

    """

    import matplotlib.pyplot as plt
    n = len(probegroup.probes)

    if same_axes:
        if 'ax' in kargs:
            ax = kargs.pop('ax')
        else:
            if probegroup.ndim == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
        axs = [ax] * n
    else:
        if 'ax' in kargs:
            raise valueError('when same_axes=False, an axes object cannot be passed into this function.')
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
        kargs['xlims'] = xlims
        kargs['ylims'] = ylims
        kargs['zlims'] = zlims
    else:
        # will be auto for each probe in each axis
        kargs['xlims'] = None
        kargs['ylims'] = None
        kargs['zlims'] = None

    kargs['title'] = False
    for i, probe in enumerate(probegroup.probes):
        plot_probe(probe, ax=axs[i], **kargs)


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
        xlims, ylims, zlims =  lims, lims, lims


    return xlims, ylims, zlims

