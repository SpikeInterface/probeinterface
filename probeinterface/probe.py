import numpy as np

_possible_electrode_shapes = ['circle', 'square', 'rect']


class Probe:
    """
    Class to handle the geometry of one probe.
    
    Handle mainly electrode position.
    
    Can be 2D or 3D.
    
    Can handle also optionally the shape of eletrode and the shape of the probe.
    
    
    """
    def __init__(self, ndim=2, si_units='um'):
        """
        
        Parameters
        ----------
        ndim: 2 or 3
            handle 2D or 3D probe
        
        si_units: 'um', 'mm', 'm'
        
        """
        assert ndim in (2, 3)
        self.ndim = ndim
        self.si_units = si_units
        
        # electrode position and shape : handle with arrays
        self.electrode_positions = None
        self.electrode_plane_axes = None
        self.electrode_shapes = None
        self.electrode_shape_params = None
        
        # vertices for the shape of the probe
        self.probe_shape_vertices = None
        
        # this handle the wiring to device : channel index on device side.
        # this is due to complex routing
        self.device_channel_indices = None
        
        # the Probe can belong to a ProbeBunch
        self._probe_bunch = None
    
    def get_electrode_count(self):
        """
        Return how many electrodes on the probe.
        """
        assert self.electrode_positions is not None
        return len(self.electrode_positions)
    
    def set_electrodes(self, positions=None, plane_axes=None, shapes='circle',
                shape_params={'radius': 10}):
        """
        Parameters
        ----------
        positions :array (num_electrodes, ndim)
            Posisitions of electrodes.
        
        plane_axes:  (num_electrodes, 2, ndim)
            This defines the axes of the electrode plane (2d or 3d)
            
        shapes: scalar or array in 'circle'/'square'/'rect'
            Shape for each electrodes.
        
        shape_params dict or list of dict
            Contain kargs for shapes ("radius" for circle, "width" for sqaure, "width/height" for rect)
        """
        assert positions is not None
        
        positions = np.array(positions)
        if positions.shape[1] != self.ndim:
            raise ValueErrorr('posistions.shape[1] and ndim do not match!')
        
        self.electrode_positions = positions
        n = positions.shape[0]
        
        # This defines the electrod plane (2d or 3d) where the electrode lies.
        # For 2D we make auto
        if plane_axes is None:
            if self.ndim ==3:
                raise ValueError('you need to give plane_axes')
            else:
                plane_axes = np.zeros((n, 2, self.ndim))
                plane_axes[:, 0, 0] = 1
                plane_axes[:, 1, 1] = 1
        plane_axes = np.array(plane_axes)
        self.electrode_plane_axes = plane_axes

        # shape
        if isinstance(shapes, str):
            shapes = [shapes] * n
        shapes = np.array(shapes)
        if not np.all(np.in1d(shapes, _possible_electrode_shapes)):
            raise ValueError(f'Electrodes shape must be in {_possible_electrode_shapes}')
        if shapes.shape[0] !=n:
            raise ValueError(f'Electrodes shape must have same length as posistions')
        self.electrode_shapes = np.array(shapes)
        
        # shape params
        if isinstance(shape_params, dict):
            shape_params = [shape_params] * n
        self.electrode_shape_params = np.array(shape_params)
    
    def set_shape_vertices(self, shape_vertices):
        shape_vertices = np.asarray(shape_vertices)
        if shape_vertices.shape[1] != self.ndim:
            raise ValueErrorr('shape_vertices.shape[1] and ndim do not match!')
        self.probe_shape_vertices = shape_vertices
    
    def create_auto_shape(self, probe_type='tip', margin=20):
        if self.ndim !=2:
            raise NotImplementedError
        
        x0 = np.min(self.electrode_positions[:, 0])
        x1 = np.max(self.electrode_positions[:, 0])
        x0 -= margin
        x1 += margin
        
        y0 = np.min(self.electrode_positions[:, 1])
        y1 = np.max(self.electrode_positions[:, 1])
        y0 -= margin
        y1 += margin
        
        if probe_type == 'rect':
            vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        elif probe_type == 'tip':
            tip = ((x0+x1)*0.5, y0 - margin*4)
            vertices = [(x0, y0), tip, (x1, y0), (x1, y1), (x0, y1)]
        else:
            raise ValueError()
        self.set_shape_vertices(vertices)
    
    def set_device_channel_indices(self, channel_indices):
        """
        Set the channel indices on device side.
        
        If some channel are not connected or not recorded then channel can be "-1"
        """
        if channel_indices.size != self.get_electrode_count():
            valueError('channel_indices have not the same size as electrode')
        self.device_channel_indices = channel_indices
        if self._probe_bunch is not None:
            self._probe_bunch.check_global_device_wiring()
    
    def copy(self):
        """
        Copy to another Probe instance.
        
        Note: device_channel_indices is not copied.
        """
        other = Probe()
        other.set_electrodes(
                    positions=self.electrode_positions.copy(),
                    plane_axes=self.electrode_plane_axes.copy(),
                    shapes=self.electrode_shapes.copy(),
                    shape_params=self.electrode_shape_params.copy())
        if self.probe_shape_vertices is not None:
            other.set_shape_vertices(self.probe_shape_vertices.copy())
        # channel_indices are not copied
        return other

    def to_3d(self, plane='xz'):
        """
        Transform 2d probe to 3d probe.
        
        Note: device_channel_indices is not copied.
        
        Parameters
        ----------
        plane: 'xy', 'yz' ', xz'
        """
        assert self.ndim ==2
        
        probe3d = Probe(ndim=3, si_units=self.si_units)
        
        # electrodes
        positions = _2d_to_3d(self.electrode_positions, plane)
        plane0 = _2d_to_3d(self.electrode_plane_axes[:, 0, :], plane)
        plane1 = _2d_to_3d(self.electrode_plane_axes[:, 1, :], plane)
        plane_axes = np.concatenate([plane0[:, np.newaxis, :], plane1[:, np.newaxis, :]], axis=1)
        probe3d.set_electrodes(
                    positions=positions,
                    plane_axes=plane_axes,
                    shapes=self.electrode_shapes.copy(),
                    shape_params=self.electrode_shape_params.copy())

        # shape
        if self.probe_shape_vertices is not None:
            vertices3d = _2d_to_3d(self.probe_shape_vertices, plane)
            probe3d.set_shape_vertices(vertices3d)
        
        if self.device_channel_indices is not None:
            probe3d.device_channel_indices = self.device_channel_indices
        
        return probe3d
    
    def rotate(self, theta, origin, axis=None):
        """
        Rorate the probe the specified axis

        Parameters
        ----------
        theta
        
        origin
        
        axis: None for 2d vector for 3d
        """
        if self.ndim == 2:
            raise NotImplementedError
        elif self.ndim == 3:
            raise NotImplementedError
        
        
    
    def move(self, direction):
        """
        Move the probe toward a direction.
        
        Parameters
        ----------
        direction: array shape (2, ) or (3, )
        """
        direction = np.asarray(direction)
        assert direction.shape[0] == self.ndim
        
        self.electrode_positions += direction
        
        if self.probe_shape_vertices is not None:
            self.probe_shape_vertices += direction
        


def _2d_to_3d(data2d, plane):
    data3d = np.zeros((data2d.shape[0], 3), dtype=data2d.dtype)
    if plane == 'xy':
        data3d[:, 0] = data2d[:, 0]
        data3d[:, 1] = data2d[:, 1]
    elif plane == 'yz':
        data3d[:, 1] = data2d[:, 0]
        data3d[:, 2] = data2d[:, 1]
    elif plane == 'xz':
        data3d[:, 0] = data2d[:, 0]
        data3d[:, 2] = data2d[:, 1]
    else:
        raise ValueError('Bad plane')
    return data3d
    
    
    
    
    


