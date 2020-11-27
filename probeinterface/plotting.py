"""
A simple implementation of ploting Probe and ProbeBunch
using matplotlib.

Depending Probe.ndim the plotting is done in 2d or 3d
"""

# matplotlib is a weak dep
import numpy as np


def plot_probe_2d(probe, ax=None, electrode_colors=None, with_channel_index=False,
                    probe_shape_kwargs={ 'facecolor':'green', 'edgecolor':'k', 'lw':0.5, 'alpha':0.3}
                    ):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, Rectangle
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    n = probe.get_electrode_count()
    
    if electrode_colors is None:
        electrode_colors = [ 'orange'] * n
    
    # here we plot electrode with transparent color (alpha=0)
    # this is stupid but help for the auto lim
    x, y = probe.electrode_positions.T
    ax.scatter(x, y, s=5, marker='o', alpha=0)
    
    # electrodes
    electrode_plot_opt = dict(alpha=0.7, edgecolor=[0.3, 0.3, 0.3], lw=0.5)
    for i in range(n):
        shape = probe.electrode_shapes[i]
        shape_param = probe.electrode_shape_params[i]
        x, y = probe.electrode_positions[i]
        
        if shape == 'circle':
            patch = Circle((x, y), shape_param['radius'],
                    facecolor=electrode_colors[i], **electrode_plot_opt)
        elif shape == 'square':
            w = shape_param['width']
            patch = Rectangle((x - w/2, y - w /2), w, w,
                    facecolor=electrode_colors[i], **electrode_plot_opt)
        elif shape == 'rect':
            w = shape_param['width']
            h = shape_param['height']
            patch = Rectangle((x - w/2, y - h /2), w, h,
                    facecolor=electrode_colors[i], **electrode_plot_opt)
        else:
            raise ValueError
        
        ax.add_patch(patch)
    
    # probe shape
    vertices = probe.probe_shape_vertices
    if vertices is not None:
        poly = Polygon(vertices,**probe_shape_kwargs)
        ax.add_patch(poly)
    
    if with_channel_index:
        for i in range(n):
            x, y = probe.electrode_positions[i]
            
            if probe.device_channel_indices is None:
                txt = f'{i}'
            else:
                chan_ind = probe.device_channel_indices[i]
                txt = f'prb{i}\ndev{chan_ind}'
            ax.text(x, y, txt, ha='center', va='center')



def plot_probe_3d(probe, ax=None, electrode_colors=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Polygon, Circle, Rectangle
    import mpl_toolkits.mplot3d.art3d as art3d
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    n = probe.get_electrode_count()
    
    if electrode_colors is None:
        electrode_colors = [ 'orange'] * n
    
    # electrodes
    positions = probe.electrode_positions
    min_, max_ = np.min(positions), np.max(positions)
    
    electrode_plot_opt = dict(alpha=0.7, edgecolor=[0.3, 0.3, 0.3], lw=0.5)
    
    vertices = []
    for i in range(n):
        shape = probe.electrode_shapes[i]
        shape_param = probe.electrode_shape_params[i]
        plane_axe = probe.electrode_plane_axes[i]
        pos = probe.electrode_positions[i]
        
        if shape == 'circle':
            r = shape_param['radius']
            theta = np.linspace(0, 2 * np.pi, 360)
            theta = np.tile(theta[:, np.newaxis], [1, 3])
            one_vertice = pos + r * np.cos(theta) * plane_axe[0] + \
                                r * np.sin(theta) * plane_axe[1]
        elif shape == 'square':
            w = shape_param['width']
            one_vertice = [
                pos - w / 2 * plane_axe[0] - w / 2 * plane_axe[1],
                pos - w / 2 * plane_axe[0] + w / 2 * plane_axe[1],
                pos + w / 2 * plane_axe[0] + w / 2 * plane_axe[1],
                pos + w / 2 * plane_axe[0] - w / 2 * plane_axe[1],
            ]
        elif shape == 'rect':
            w = shape_param['width']
            h = shape_param['height']
            one_vertice = [
                pos - w / 2 * plane_axe[0] - h / 2 * plane_axe[1],
                pos - w / 2 * plane_axe[0] + h / 2 * plane_axe[1],
                pos + w / 2 * plane_axe[0] + h / 2 * plane_axe[1],
                pos + w / 2 * plane_axe[0] - h / 2 * plane_axe[1],
            ]
        else:
            raise ValueError            
        vertices.append(one_vertice)
    
    poly3d = Poly3DCollection(vertices,color=electrode_colors,  **electrode_plot_opt)
    ax.add_collection3d(poly3d)
    
    # probe shape
    vertices = probe.probe_shape_vertices
    if vertices is not None:
        poly = Poly3DCollection([vertices], facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
        ax.add_collection3d(poly)
        
        min_, max_ = np.min(vertices), np.max(vertices)

    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)
    ax.set_zlim(min_, max_)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_probe(probe, ax=None, **kargs):
    if probe.ndim == 2:
        plot_probe_2d(probe, ax=ax, **kargs)
    elif probe.ndim == 3:
        plot_probe_3d(probe, ax=ax, **kargs)


def plot_probe_bunch(probebunch, same_axe=True, **kargs):
    import matplotlib.pyplot as plt
    n = len(probebunch.probes)
    
    if same_axe:
        if probebunch.ndim == 2:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        axs = [ax] * n
    else:
        if probebunch.ndim == 2:
            fig, axs = plt.subplots(ncols=n, nrows=1)
            if n==1:
                axs = [axs]
        else:
            raise NotImplementedError

    for i, probe in enumerate(probebunch.probes):
        plot_probe(probe, ax=axs[i], **kargs)

