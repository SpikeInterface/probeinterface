"""
A very first draft of plotting in 2d.

"""

# matplotlib is a weak dep



def plot_probe_2d(probe, ax=None, electrode_colors=None):
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
            raise NotImplementedError
        else:
            raise ValueError
        
        ax.add_patch(patch)
    
    vertices = probe.probe_shape_vertices
    if vertices is not None:
        poly = Polygon(vertices, facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
        ax.add_patch(poly)
    
    

def plot_probe_3d(probe, ax=None):
    import matplotlib.pyplot as plt
    raise NotImplementedError

def plot_probe(probe, ax=None):
    if probe.ndim == 2:
        plot_probe_2d(probe, ax=ax)
    elif probe.ndim == 3:
        plot_probe_3d(probe, ax=ax)



def plot_probe_bunch_2d(probebunch, separate_axes=True):
    import matplotlib.pyplot as plt
    n = len(probebunch.probes)
    
    if separate_axes:
        fig, axs = plt.subplots(ncols=n, nrows=1)
    else:
        fig, ax = plt.subplots()
        axs = [ax] * n
    
    for i, probe in enumerate(probebunch.probes):
        plot_probe(probe, ax=axs[i])


def plot_probe_bunch_3d(probebunch):
    import matplotlib.pyplot as plt
    raise NotImplementedError
    

def plot_probe_bunch(probebunch, **kargs):
    #TODO
    plot_probe_bunch_2d(probebunch, **kargs)
    
    
    
    
    
    
    