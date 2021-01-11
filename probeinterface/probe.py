import numpy as np

from .shank import Shank

_possible_electrode_shapes = ['circle', 'square', 'rect']


class Probe:
    """
    Class to handle the geometry of one probe.
    
    This class mainly handles electrode positions, in 2D or 3D. Optionally, it can also handle the shape of the
    electrodes and the shape of the probe.
    
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
        self.ndim = int(ndim)
        self.si_units = str(si_units)

        # electrode position and shape : handle with arrays
        self.electrode_positions = None
        self.electrode_plane_axes = None
        self.electrode_shapes = None
        self.electrode_shape_params = None

        # vertices for the shape of the probe
        self.probe_planar_contour = None

        # This handle shankd ids
        self.shank_ids = None

        # this handle the wiring to device : channel index on device side.
        # this is due to complex routing
        #  This must be unique at Probe AND ProbeGroup level
        self.device_channel_indices = None

        # Handle ids with str so it can be displayed like names
        #  This must be unique at Probe AND ProbeGroup level
        self.electrode_ids = None
        
        # annotation:  a dict that contain all meta information about 
        # the probe (name, manufacturor, date of production, ...)
        # See
        self.annotations = dict(name='')

        # the Probe can belong to a ProbeGroup
        self._probe_group = None
    
    def get_title(self):
        if self.electrode_positions is None:
            txt = 'Undefined probe'
        else:
            n = self.get_electrode_count()
            name = self.annotations.get('name', '')
            manufacturer = self.annotations.get('manufacturer', '')
            if len(name) >0 or len(manufacturer):
                txt = f'{manufacturer} - {name} - {n}ch'
            else:
                txt = f'Probe - {n}ch'
        return txt

    def __repr__(self):
        return self.get_title()
    
    def annotate(self, **kwargs):
        self.annotations.update(kwargs)
        self.check_annotations()
    
    def check_annotations(self):
        d = self.annotations
        if 'first_index' in d:
            assert d['first_index'] in (0, 1)
    
    def get_electrode_count(self):
        """
        Return the number of electrodes on the probe.
        """
        assert self.electrode_positions is not None
        return len(self.electrode_positions)
    
    def get_shank_count(self):
        """
        Return  the number of shank for this probe
        """
        assert self.shank_ids is not None
        n = len(np.unique(self.shank_ids))
        return n

    def set_electrodes(self, positions=None, 
                    shapes='circle', shape_params={'radius': 10},
                    plane_axes=None, shank_ids=None):
        """
        Parameters
        ----------
        positions :array (num_electrodes, ndim)
            Posisitions of electrodes.
        
        shapes: scalar or array in 'circle'/'square'/'rect'
            Shape for each electrodes.
        
        shape_params dict or list of dict
            Contain kargs for shapes ("radius" for circle, "width" for sqaure, "width/height" for rect)
        plane_axes:  (num_electrodes, 2, ndim)
            This defines the axes of the electrode plane (2d or 3d)
        shank_ids: None or vector of int
            This define the shank id for electrodes. If None then
            there are assign to a unique Shank.
        """
        assert positions is not None

        positions = np.array(positions)
        if positions.shape[1] != self.ndim:
            raise ValueError('posistions.shape[1] and ndim do not match!')

        self.electrode_positions = positions
        n = positions.shape[0]

        # This defines the electrod plane (2d or 3d) where the electrode lies.
        # For 2D we make auto
        if plane_axes is None:
            if self.ndim == 3:
                raise ValueError('you need to give plane_axes')
            else:
                plane_axes = np.zeros((n, 2, self.ndim))
                plane_axes[:, 0, 0] = 1
                plane_axes[:, 1, 1] = 1
        plane_axes = np.array(plane_axes)
        self.electrode_plane_axes = plane_axes

        if shank_ids is None:
            self.shank_ids = np.zeros(n, dtype='int64')
        else:
            self.shank_ids = np.asarray(shank_ids)
            if self.shank_ids.size != n:
                raise ValueError('shan_ids have wring size') 

        # shape
        if isinstance(shapes, str):
            shapes = [shapes] * n
        shapes = np.array(shapes)
        if not np.all(np.in1d(shapes, _possible_electrode_shapes)):
            raise ValueError(f'Electrodes shape must be in {_possible_electrode_shapes}')
        if shapes.shape[0] != n:
            raise ValueError(f'Electrodes shape must have same length as posistions')
        self.electrode_shapes = np.array(shapes)

        # shape params
        if isinstance(shape_params, dict):
            shape_params = [shape_params] * n
        self.electrode_shape_params = np.array(shape_params)

    def set_planar_contour(self, contour_polygon):
        contour_polygon = np.asarray(contour_polygon)
        if contour_polygon.shape[1] != self.ndim:
            raise ValueError('contour_polygon.shape[1] and ndim do not match!')
        self.probe_planar_contour = contour_polygon

    def create_auto_shape(self, probe_type='tip', margin=20):
        if self.ndim != 2:
            raise ValueError('Auto shape is supported only for 2d')

        x0 = np.min(self.electrode_positions[:, 0])
        x1 = np.max(self.electrode_positions[:, 0])
        x0 -= margin
        x1 += margin

        y0 = np.min(self.electrode_positions[:, 1])
        y1 = np.max(self.electrode_positions[:, 1])
        y0 -= margin
        y1 += margin

        if probe_type == 'rect':
            polygon = [(x0, y1), (x0, y0), (x1, y0), (x1, y1), ]
        elif probe_type == 'tip':
            tip = ((x0 + x1) * 0.5, y0 - margin * 4)
            polygon = [(x0, y1), (x0, y0), tip, (x1, y0), (x1, y1), ]
        else:
            raise ValueError()
        self.set_planar_contour(polygon)

    def set_device_channel_indices(self, channel_indices):
        """
        Set manually the channel indices on device side.
        
        If some channel are not connected or not recorded then channel can be "-1"


        Parameters
        ----------
        channel_indices: array of int
        
        """
        channel_indices = np.asarray(channel_indices)
        if channel_indices.size != self.get_electrode_count():
            valueError('channel_indices have not the same size as electrode')
        self.device_channel_indices = channel_indices
        if self._probe_group is not None:
            self._probe_group.check_global_device_wiring_and_ids()
    
    def wiring_to_device(self, pathway, channel_offset=0):
        """
        Automatically device_channel_indices.
        
        This use internal 
        
        See probeinterface.wiring module.
        
        Parameters
        ----------
        
        pathway: str
           For instance 'H32>RHD'
        
        
        """
        from .wiring import wire_probe
        wire_probe(self, pathway, channel_offset=channel_offset)

    
    def set_electrode_ids(self, elec_ids):
        """
        Set electrode ids. This is handle with string.
        It is like a name but must be **unique** for the Probe
        and also for the **ProbeGroup**
        
        Parameters
        ----------
        elec_ids: array of str
            If elec_ids is int or float then convert to str
        """
        elec_ids = np.asarray(elec_ids)

        if elec_ids.size != self.get_electrode_count():
            ValueError('channel_indices have not the same size as electrode')

        if elec_ids.dtype.kind != 'U':
            elec_ids = elec_ids.astype('U')

        self.electrode_ids = elec_ids
        if self._probe_group is not None:
            self._probe_group.check_global_device_wiring_and_ids()

    def set_shank_ids(self, shank_ids):
        """
        Set shank ids
        """
        shank_ids = np.asarray(shank_ids)
        if self.shank_ids.size != self.get_electrode_count():
            raise ValueError('shan_ids have wring size') 
        self.shank_ids = shank_ids

    def get_shanks(self):
        """
        Return the list of Shank object for this Probe
        """
        assert self.shank_ids is not None
        shanks = []
        for shank_id  in np.unique(self.shank_ids):
            shank = Shank(self, shank_id)
            shanks.append(shank)
        return shanks

    def copy(self):
        """
        Copy to another Probe instance.
        
        Note: device_channel_indices is not copied
        and electrode_ids is not copied
        """
        other = Probe()
        other.set_electrodes(
            positions=self.electrode_positions.copy(),
            plane_axes=self.electrode_plane_axes.copy(),
            shapes=self.electrode_shapes.copy(),
            shape_params=self.electrode_shape_params.copy())
        if self.probe_planar_contour is not None:
            other.set_planar_contour(self.probe_planar_contour.copy())
        # channel_indices are not copied
        return other

    def to_3d(self, plane='xz'):
        """
        Transform 2d probe to 3d probe.
        
        Note: device_channel_indices is not copied.
        
        Parameters
        ----------
        plane: str
            'xy', 'yz' ', xz'

        """
        assert self.ndim == 2

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
        if self.probe_planar_contour is not None:
            vertices3d = _2d_to_3d(self.probe_planar_contour, plane)
            probe3d.set_planar_contour(vertices3d)

        if self.device_channel_indices is not None:
            probe3d.device_channel_indices = self.device_channel_indices

        return probe3d

    def get_electrodes_vertices(self):
        """
        return a list of electrodes vertices.
        """
        vertices = []
        for i in range(self.get_electrode_count()):
            shape = self.electrode_shapes[i]
            shape_param = self.electrode_shape_params[i]
            plane_axe = self.electrode_plane_axes[i]
            pos = self.electrode_positions[i]

            if shape == 'circle':
                r = shape_param['radius']
                theta = np.linspace(0, 2 * np.pi, 360)
                theta = np.tile(theta[:, np.newaxis], [1, self.ndim])
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
        return vertices

    def move(self, translation_vetor):
        """
        Move the probe toward a direction.
        
        Parameters
        ----------
        translation_vetor: array shape (2, ) or (3, )
        """
        translation_vetor = np.asarray(translation_vetor)
        assert translation_vetor.shape[0] == self.ndim

        self.electrode_positions += translation_vetor

        if self.probe_planar_contour is not None:
            self.probe_planar_contour += translation_vetor

    def rotate(self, theta, center=None, axis=None):
        """
        Rorate the probe the specified axis

        Parameters
        ----------
        theta: float
            In degree, anticlockwise.
        
        center: center of rotation
            If None the center of probe is take
        
        axis: None for 2d vector for 3d
        """
        if center is None:
            center = np.mean(self.electrode_positions, axis=0)

        center = np.asarray(center)
        assert center.size == self.ndim
        center = center[None, :]

        theta = np.deg2rad(theta)

        if self.ndim == 2:
            if axis is not None:
                raise ValueError('axis must be None for 2d')
            R = _rotation_matrix_2d(theta)
        elif self.ndim == 3:
            R = _rotation_matrix_3d(axis, theta).T

        new_positions = (self.electrode_positions - center) @ R + center

        new_plane_axes = np.zeros_like(self.electrode_plane_axes)
        for i in range(2):
            new_plane_axes[:, i, :] = (self.electrode_plane_axes[:, i,
                                       :] - center + self.electrode_positions) @ R + center - new_positions

        self.electrode_positions = new_positions
        self.electrode_plane_axes = new_plane_axes

        if self.probe_planar_contour is not None:
            new_vertices = (self.probe_planar_contour - center) @ R + center
            self.probe_planar_contour = new_vertices

    def rotate_electrodes(self, thetas, center=None, axis=None):
        """
        Rotate each electrodes.
        Internaly modify the electrode_plane_axes.
        
        Parameters
        ----------
        thetas: array of float
            rotation angle in degree.
            If scalar then it is applied to all electrodes.
        """
        if self.ndim == 3:
            raise ValueError('By electrode rotation is implemented only for 2d')

        n = self.get_electrode_count()

        if isinstance(thetas, (int, float)):
            thetas = np.array([thetas] * n, dtype='float64')

        thetas = np.deg2rad(thetas)

        for e in range(n):
            R = _rotation_matrix_2d(thetas[e])
            for i in range(2):
                self.electrode_plane_axes[e, i, :] = self.electrode_plane_axes[e, i, :] @ R

    _dump_attr_names = ['ndim', 'si_units', 'annotations',
                        'electrode_positions', 'electrode_plane_axes',
                        'electrode_shapes', 'electrode_shape_params',
                        'probe_planar_contour', 'device_channel_indices', 'electrode_ids',
                        'shank_ids']

    def to_dict(self, array_as_list=False):
        """
        Create a dict of all necessary attributes.
        Usefull for dumping or saving to hdf5.
        """
        d = {}
        for k in self._dump_attr_names:
            v = getattr(self, k, None)
            if array_as_list and v is not None and isinstance(v, np.ndarray): 
                v = v.tolist()
            print(k, v)
            if v is not None:
                d[k] = v
        return d

    @staticmethod
    def from_dict(d):
        probe = Probe(ndim=d['ndim'], si_units=d['si_units'])

        probe.set_electrodes(
            positions=d['electrode_positions'],
            plane_axes=d['electrode_plane_axes'],
            shapes=d['electrode_shapes'],
            shape_params=d['electrode_shape_params'])

        v = d.get('probe_planar_contour', None)
        if v is not None:
            probe.set_planar_contour(v)

        v = d.get('device_channel_indices', None)
        if v is not None:
            probe.set_device_channel_indices(v)
        
        probe.annotate(**d['annotations'])

        return probe
    
    def to_dataframe(self):
        import pandas as pd
        index = np.arange(self.get_electrode_count(), dtype=int)
        df = pd.DataFrame(index=index)
        df['x'] = self.electrode_positions[:, 0]
        df['y'] = self.electrode_positions[:, 1]
        if self.ndim == 3:
            df['z'] = self.electrode_positions[:, 2]
        df['electrode_shapes'] = self.electrode_shapes
        for i, p in enumerate(self.electrode_shape_params):
            for k, v in p.items():
                df.at[i, k] = v
        df['device_channel_indices'] = self.device_channel_indices
        df['shank_ids'] = self.shank_ids
        
        
        return df


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


def _rotation_matrix_2d(theta):
    """
    Returns 2D rotation matrix

    Parameters
    ----------
    theta: float
        Angle in radians for rotation anti-clockwise

    Returns
    -------
    R: np.array
        2D rotation matrix
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R


def _rotation_matrix_3d(axis, theta):
    '''
    Returns 3D rotation matrix
    
    Copy/paste from MEAutility

    Parameters
    ----------
    axis: np.array or list
        3D axis of rotation
    theta: float
        Angle in radians for rotation anti-clockwise

    Returns
    -------
    R: np.array
        3D rotation matrix
    '''
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                  [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                  [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return R
