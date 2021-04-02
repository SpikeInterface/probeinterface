import numpy as np

from .shank import Shank

_possible_contact_shapes = ['circle', 'square', 'rect']


class Probe:
    """
    Class to handle the geometry of one probe.

    This class mainly handles contact positions, in 2D or 3D.
    Optionally, it can also handle the shape of the
    contacts and the shape of the probe.

    """

    def __init__(self, ndim=2, si_units='um'):
        """
        Some attributes are protected and have to be set with setters:
          * set_contacts(...)
          * set_shank_ids

        Parameters
        ----------
        ndim: 2 or 3
            handle 2D or 3D probe

        si_units: 'um', 'mm', 'm'

        """

        assert ndim in (2, 3)
        self.ndim = int(ndim)
        self.si_units = str(si_units)

        # contact position and shape : handle with arrays

        self._contact_positions = None
        self._contact_plane_axes = None
        self._contact_shapes = None
        self._contact_shape_params = None

        # vertices for the shape of the probe
        self.probe_planar_contour = None

        # This handles the shank id per contact
        self._shank_ids = None

        # This handles the wiring to device : channel index on device side.
        # this is due to complex routing
        #  This must be unique at Probe AND ProbeGroup level
        self.device_channel_indices = None

        # Handle ids with str so it can be displayed like names
        #  This must be unique at Probe AND ProbeGroup level
        self._contact_ids = None

        # annotation:  a dict that contains all meta information about
        # the probe (name, manufacturor, date of production, ...)
        # See
        self.annotations = dict(name='')

        # the Probe can belong to a ProbeGroup
        self._probe_group = None

    @property
    def contact_positions(self):
            return self._contact_positions

    @property
    def contact_plane_axes(self):
            return self._contact_plane_axes

    @property
    def contact_shapes(self):
            return self._contact_shapes

    @property
    def contact_shape_params(self):
            return self._contact_shape_params

    @property
    def contact_ids(self):
            return self._contact_ids

    @property
    def shank_ids(self):
            return self._shank_ids

    def get_title(self):
        if self.contact_positions is None:
            txt = 'Undefined probe'
        else:
            n = self.get_contact_count()
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

    def get_contact_count(self):
        """
        Return the number of contacts on the probe.
        """
        assert self.contact_positions is not None
        return len(self.contact_positions)

    def get_shank_count(self):
        """
        Return  the number of shanks for this probe
        """
        assert self.shank_ids is not None
        n = len(np.unique(self.shank_ids))
        return n

    def set_contacts(self, positions=None,
                    shapes='circle', shape_params={'radius': 10},
                    plane_axes=None, shank_ids=None):
        """
        Parameters
        ----------
        positions : array (num_contacts, ndim)
            Positions of contacts.
        shapes : scalar or array in 'circle'/'square'/'rect'
            Shape of each contact.
        shape_params : dict or list of dict
            Contains kwargs for shapes ("radius" for circle, "width" for square, "width/height" for rect)
        plane_axes : (num_contacts, 2, ndim)
            This defines the axes of the contact plane (2d or 3d)
        shank_ids : None or vector of str
            This define the shank id for contacts. If None, then
            these are assigned to a unique Shank.

        """
        assert positions is not None

        positions = np.array(positions)
        if positions.shape[1] != self.ndim:
            raise ValueError('posistions.shape[1] and ndim do not match!')

        self._contact_positions = positions
        n = positions.shape[0]

        # This defines the contact plane (2D or 3D) along which the contacts lie.
        # For 2D we make auto
        if plane_axes is None:
            if self.ndim == 3:
                raise ValueError('you need to give plane_axes')
            else:
                plane_axes = np.zeros((n, 2, self.ndim))
                plane_axes[:, 0, 0] = 1
                plane_axes[:, 1, 1] = 1
        plane_axes = np.array(plane_axes)
        self._contact_plane_axes = plane_axes

        if shank_ids is None:
            self._shank_ids = np.zeros(n, dtype=str)
        else:
            self._shank_ids = np.asarray(shank_ids).astype(str)
            if self.shank_ids.size != n:
                raise ValueError('shan_ids have wring size')

        # shape
        if isinstance(shapes, str):
            shapes = [shapes] * n
        shapes = np.array(shapes)
        if not np.all(np.in1d(shapes, _possible_contact_shapes)):
            raise ValueError(f'contacts shape must be in {_possible_contact_shapes}')
        if shapes.shape[0] != n:
            raise ValueError(f'contacts shape must have same length as posistions')
        self._contact_shapes = np.array(shapes)

        # shape params
        if isinstance(shape_params, dict):
            shape_params = [shape_params] * n
        self._contact_shape_params = np.array(shape_params)

    def set_planar_contour(self, contour_polygon):
        contour_polygon = np.asarray(contour_polygon)
        if contour_polygon.shape[1] != self.ndim:
            raise ValueError('contour_polygon.shape[1] and ndim do not match!')
        self.probe_planar_contour = contour_polygon

    def create_auto_shape(self, probe_type='tip', margin=20):
        if self.ndim != 2:
            raise ValueError('Auto shape is supported only for 2d')

        x0 = np.min(self.contact_positions[:, 0])
        x1 = np.max(self.contact_positions[:, 0])
        x0 -= margin
        x1 += margin

        y0 = np.min(self.contact_positions[:, 1])
        y1 = np.max(self.contact_positions[:, 1])
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
        Manually set the channel indices on the device side.

        If some channel are not connected or not recorded then channel can be "-1"


        Parameters
        ----------
        channel_indices : array of int

        """
        channel_indices = np.asarray(channel_indices, dtype=int)
        if channel_indices.size != self.get_contact_count():
            ValueError('channel_indices have not the same size as contact')
        self.device_channel_indices = channel_indices
        if self._probe_group is not None:
            self._probe_group.check_global_device_wiring_and_ids()

    def wiring_to_device(self, pathway, channel_offset=0):
        """
        Automatically set device_channel_indices.

        For internal use only.

        See probeinterface.wiring module.

        Parameters
        ----------

        pathway : str
           For instance 'H32>RHD'

        """

        from .wiring import wire_probe
        wire_probe(self, pathway, channel_offset=channel_offset)


    def set_contact_ids(self, contact_ids):
        """
        Set contact ids. This is handled with a string.
        It is like a name but must be **unique** for the Probe
        and also for the **ProbeGroup**

        Parameters
        ----------
        contact_ids : array of str
            If contact_ids is int or float then convert to str

        """

        contact_ids = np.asarray(contact_ids)

        if contact_ids.size != self.get_contact_count():
            ValueError('channel_indices have not the same size as contact')

        if contact_ids.dtype.kind != 'U':
            contact_ids = contact_ids.astype('U')

        self._contact_ids = contact_ids
        if self._probe_group is not None:
            self._probe_group.check_global_device_wiring_and_ids()

    def set_shank_ids(self, shank_ids):
        """
        Set shank ids
        """
        shank_ids = np.asarray(shank_ids).astype(str)
        if shank_ids.size != self.get_contact_count():
            raise ValueError(f'shank_ids have wrong size. Has to match number '
                             f'of contacts: {self.get_contact_count()}')
        self._shank_ids = shank_ids

    def get_shanks(self):
        """
        Return the list of Shank objects for this Probe
        """
        assert self.shank_ids is not None
        shanks = []
        for shank_id in np.unique(self.shank_ids):
            shank = Shank(self, shank_id)
            shanks.append(shank)
        return shanks

    def copy(self):
        """
        Copy to another Probe instance.

        Note: device_channel_indices is not copied
        and contact_ids is not copied
        """
        other = Probe()
        other.set_contacts(
            positions=self.contact_positions.copy(),
            plane_axes=self.contact_plane_axes.copy(),
            shapes=self.contact_shapes.copy(),
            shape_params=self.contact_shape_params.copy())
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
        plane : str
            'xy', 'yz' ', xz'

        """

        assert self.ndim == 2

        probe3d = Probe(ndim=3, si_units=self.si_units)

        # contacts
        positions = _2d_to_3d(self.contact_positions, plane)
        plane0 = _2d_to_3d(self.contact_plane_axes[:, 0, :], plane)
        plane1 = _2d_to_3d(self.contact_plane_axes[:, 1, :], plane)
        plane_axes = np.concatenate([plane0[:, np.newaxis, :], plane1[:, np.newaxis, :]], axis=1)
        probe3d.set_contacts(
            positions=positions,
            plane_axes=plane_axes,
            shapes=self.contact_shapes.copy(),
            shape_params=self.contact_shape_params.copy())

        # shape
        if self.probe_planar_contour is not None:
            vertices3d = _2d_to_3d(self.probe_planar_contour, plane)
            probe3d.set_planar_contour(vertices3d)

        if self.device_channel_indices is not None:
            probe3d.device_channel_indices = self.device_channel_indices

        return probe3d

    def get_contact_vertices(self):
        """
        return a list of contact vertices.
        """

        vertices = []
        for i in range(self.get_contact_count()):
            shape = self.contact_shapes[i]
            shape_param = self.contact_shape_params[i]
            plane_axe = self.contact_plane_axes[i]
            pos = self.contact_positions[i]

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

    def move(self, translation_vector):
        """
        Translate the probe in one direction.

        Parameters
        ----------
        translation_vector : array shape (2, ) or (3, )

        """

        translation_vector = np.asarray(translation_vector)
        assert translation_vector.shape[0] == self.ndim

        self._contact_positions += translation_vector

        if self.probe_planar_contour is not None:
            self.probe_planar_contour += translation_vector

    def rotate(self, theta, center=None, axis=None):
        """
        Rotate the probe around a specified axis

        Parameters
        ----------
        theta : float
            In degrees, anticlockwise.

        center : center of rotation
            If None the center of probe is used

        axis : None for 2D vector for 3D

        """

        if center is None:
            center = np.mean(self.contact_positions, axis=0)

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

        new_positions = (self.contact_positions - center) @ R + center

        new_plane_axes = np.zeros_like(self.contact_plane_axes)
        for i in range(2):
            new_plane_axes[:, i, :] = (self.contact_plane_axes[:, i,
                                       :] - center + self.contact_positions) @ R + center - new_positions

        self._contact_positions = new_positions
        self._contact_plane_axes = new_plane_axes

        if self.probe_planar_contour is not None:
            new_vertices = (self.probe_planar_contour - center) @ R + center
            self.probe_planar_contour = new_vertices

    def rotate_contacts(self, thetas, center=None, axis=None):
        """
        Rotate each contact of the probe.
        Internaly modify the contact_plane_axes.

        Parameters
        ----------
        thetas : array of float
            rotation angle in degree.
            If scalar then it is applied to all contacts.

        """

        if self.ndim == 3:
            raise ValueError('By contact rotation is implemented only for 2d')

        n = self.get_contact_count()

        if isinstance(thetas, (int, float)):
            thetas = np.array([thetas] * n, dtype='float64')

        thetas = np.deg2rad(thetas)

        for e in range(n):
            R = _rotation_matrix_2d(thetas[e])
            for i in range(2):
                self.contact_plane_axes[e, i, :] = self.contact_plane_axes[e, i, :] @ R

    _dump_attr_names = ['ndim', 'si_units', 'annotations',
                        '_contact_positions', '_contact_plane_axes',
                        '_contact_shapes', '_contact_shape_params',
                        'probe_planar_contour', 'device_channel_indices',
                        '_contact_ids', '_shank_ids']

    def to_dict(self, array_as_list=False):
        """
        Create a dict of all necessary attributes.
        Useful for dumping or saving to hdf5.
        """

        d = {}
        for k in self._dump_attr_names:
            v = getattr(self, k, None)
            if array_as_list and v is not None and isinstance(v, np.ndarray):
                v = v.tolist()
            if v is not None:
                if k.startswith('_'):
                    d[k[1:]] = v
                else:
                    d[k] = v
        return d

    @staticmethod
    def from_dict(d):
        probe = Probe(ndim=d['ndim'], si_units=d['si_units'])

        probe.set_contacts(
            positions=d['contact_positions'],
            plane_axes=d['contact_plane_axes'],
            shapes=d['contact_shapes'],
            shape_params=d['contact_shape_params'])

        v = d.get('probe_planar_contour', None)
        if v is not None:
            probe.set_planar_contour(v)

        v = d.get('device_channel_indices', None)
        if v is not None:
            probe.set_device_channel_indices(v)

        v = d.get('shank_ids', None)
        if v is not None:
            probe.set_shank_ids(v)

        v = d.get('contact_ids', None)
        if v is not None:
            probe.set_contact_ids(v)

        probe.annotate(**d['annotations'])

        return probe

    def to_numpy(self, complete=False):
        """
        Export to a numpy vector (struct array).
        This vector handle all contact attributes.

        Equivalent of to_dataframe but without pandas dependency.

        Very useful to export/slice/attach to a recording.

        Parameters
        ----------

        complete : bool
            If true, export complete information about the probe,
            including contact_plane_axes/si_units/device_channel_indices

        returns
        ---------
        arr : numpy.array
            With complex dtype
        """

        dtype = [('x', 'float64'), ('y', 'float64')]
        if self.ndim == 3:
            dtype += [('z', 'float64')]
        dtype += [('contact_shapes', 'U64')]
        param_shape = []
        for i, p in enumerate(self.contact_shape_params):
            for k, v in p.items():
                if k not in param_shape:
                    param_shape.append(k)
        for k in param_shape:
            dtype += [(k, 'float64')]
        dtype += [('shank_ids', 'U64'), ('contact_ids', 'U64')]

        if complete:
            dtype += [('device_channel_indices', 'int64')]
            dtype += [('si_units', 'U64')]
            for i in range(self.ndim):
                dim = ['x', 'y', 'z'][i]
                dtype += [(f'plane_axis_{dim}_0', 'float64')]
                dtype += [(f'plane_axis_{dim}_1', 'float64')]

        arr = np.zeros(self.get_contact_count(), dtype=dtype)
        arr['x'] = self.contact_positions[:, 0]
        arr['y'] = self.contact_positions[:, 1]
        if self.ndim == 3:
            arr['z'] = self.contact_positions[:, 2]
        arr['contact_shapes'] = self.contact_shapes
        for i, p in enumerate(self.contact_shape_params):
            for k, v in p.items():
                arr[k][i] = v

        arr['shank_ids'] = self.shank_ids

        if self.contact_ids is None:
            arr['contact_ids'] = [''] * self.get_contact_count()
        else:
            arr['contact_ids'] = self.contact_ids

        if complete:
            arr['si_units'] = self.si_units

            #(num_contacts, 2, ndim)
            for i in range(self.ndim):
                dim = ['x', 'y', 'z'][i]
                arr[f'plane_axis_{dim}_0'] = self.contact_plane_axes[:, 0, i]
                arr[f'plane_axis_{dim}_1'] = self.contact_plane_axes[:, 1, i]

            if self.device_channel_indices is None:
               arr['device_channel_indices'] = -1
            else:
                arr['device_channel_indices'] = self.device_channel_indices

        return arr

    @staticmethod
    def from_numpy(arr):
        """
        Create Probe from a complex numpy array
        see Probe.to_numpy()
        """

        fields = list(arr.dtype.fields)

        if 'z' in fields:
            ndim = 3
        else:
            ndim = 2

        assert 'x' in fields
        assert 'y' in fields
        if 'si_units' in fields:
            assert np.unique(arr['si_units']).size == 1
            si_units = np.unique(arr['si_units'])[0]
        else:
            si_units = 'um'
        probe = Probe(ndim=ndim, si_units=si_units)

        # contacts
        positions = np.zeros((arr.size, ndim), dtype='float64')
        for i, dim in enumerate(['x', 'y', 'z'][:ndim]):
            positions[:, i] = arr[dim]
        shapes = arr['contact_shapes']
        shape_params = []
        for i, shape in enumerate(shapes):
            if shape == 'circle':
                p = {'radius': float(arr['radius'][i])}
            elif shape == 'square':
                p = {'width': float(arr['width'][i])}
            elif shape == 'rect':
                p = {'width': float(arr['width'][i]),
                     'height': float(arr['height'][i])}
            else:
                raise ValueError('You are in bad shape')
            shape_params.append(p)

        if 'plane_axis_x_0' in fields:
            #(num_contacts, 2, ndim)
            plane_axes = np.zeros((arr.size, 2, ndim))
            for i in range(ndim):
                dim = ['x', 'y', 'z'][i]
                plane_axes[:, 0, i] = arr[f'plane_axis_{dim}_0']
                plane_axes[:, 1, i] = arr[f'plane_axis_{dim}_1']
        else:
            plane_axes = None

        probe.set_contacts(
            positions=positions,
            plane_axes=plane_axes,
            shapes=shapes,
            shape_params=shape_params)

        if 'device_channel_indices' in fields:
            dev_channel_indices = arr['device_channel_indices']
            probe.set_device_channel_indices(dev_channel_indices)
        if 'shank_ids' in fields:
            probe.set_shank_ids(arr['shank_ids'])
        if 'contact_ids' in fields:
            probe.set_contact_ids(arr['contact_ids'])

        return probe

    def to_dataframe(self, complete=False):
        """
        Export the probe to a pandas dataframe

        Parameters
        ----------
        complete : bool
            If true, export complete information about the probe,
            including the probe plane axis.

        """

        import pandas as pd
        arr = self.to_numpy(complete=complete)
        df = pd.DataFrame.from_records(arr)
        df.index = np.arange(df.shape[0], dtype='int64')
        return df

    @staticmethod
    def from_dataframe(df):
        arr = df.to_records(index=False)
        return Probe.from_numpy(arr)

    def to_image(self, values, pixel_size=0.5, num_pixel=None, method='linear',
                 xlims=None, ylims=None):
        """
        Generated a 2d (image) from a values vector which an interpolation
        into a grid mesh.


        Parameters
        ----------
        values :
            vector same size as contact number to be color plotted
        pixel_size :
            size of one pixel in micrometers
        num_pixel :
            alternative to pixel_size give pixel number of the image width
        method : 'linear' or 'nearest' or 'cubic'
        xlims : tuple or None
            Force image xlims
        ylims : tuple or None
            Force image ylims

        returns
        --------
        image : 2d array
        xlims :
        ylims :

        """
        from scipy.interpolate import griddata
        assert self.ndim == 2
        assert values.shape == (self.get_contact_count(), ), 'Bad boy: values must have size equal contact count'

        if xlims is None:
            x0 = np.min(self.contact_positions[:, 0])
            x1 = np.max(self.contact_positions[:, 0])
            xlims = (x0, x1)

        if ylims is None:
            y0 = np.min(self.contact_positions[:, 1])
            y1 = np.max(self.contact_positions[:, 1])
            ylims = (y0, y1)

        x0, x1 = xlims
        y0, y1= ylims

        if num_pixel is not None:
            pixel_size = (x1 - x0) / num_pixel


        grid_x, grid_y = np.meshgrid(np.arange(x0, x1, pixel_size), np.arange(y0, y1, pixel_size))
        image = griddata(self.contact_positions, values, (grid_x, grid_y), method=method)

        if method == 'nearest':
            # hack to force nan when nereast to avoid interpolation in the full rectangle
            image2, _, _ = self.to_image(values, pixel_size=pixel_size,method='linear', xlims=xlims, ylims=ylims)
            #~ print(im
            image[np.isnan(image2)] = np.nan

        return image, xlims, ylims

    def get_slice(self, selection):
        """
        Get a copy of the Probe with a sub selection of contacts.

        Selection can be boolean or by index

        Parameters
        ----------
        selection : np.array of bool or int (for index)

        """

        n = self.get_contact_count()

        selection = np.asarray(selection)
        if selection.dtype =='bool':
            assert selection.shape == (n, )
        elif selection.dtype.kind =='i':
            assert np.unique(selection).size == selection.size
            assert 0 <= np.min(selection) < n
            assert 0 <= np.max(selection) < n
        else:
            raise ValueError('selection must be bool array or int array')


        d = self.to_dict(array_as_list=False)
        for k, v in d.items():
            if k == 'probe_planar_contour':
                continue

            if isinstance(v, np.ndarray):
                assert v.shape[0] == n
                d[k] = v[selection].copy()

        sliced_probe = Probe.from_dict(d)

        return sliced_probe





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
    theta : float
        Angle in radians for rotation (anti-clockwise)

    Returns
    -------
    R : np.array
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
    axis : np.array or list
        3D axis of rotation
    theta : float
        Angle in radians for rotation anti-clockwise

    Returns
    -------
    R : np.array
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
