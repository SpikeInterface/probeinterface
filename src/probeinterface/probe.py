from __future__ import annotations
import numpy as np
from typing import Optional
from pathlib import Path

from .shank import Shank

_possible_contact_shapes = ["circle", "square", "rect"]


class Probe:
    """
    Class to handle the geometry of one probe.

    This class mainly handles contact positions, in 2D or 3D.
    Optionally, it can also handle the shape of the
    contacts and the shape of the probe.

    """

    def __init__(
        self,
        ndim: int = 2,
        si_units: str = "um",
        name: Optional[str] = None,
        serial_number: Optional[str] = None,
        model_name: Optional[str] = None,
        manufacturer: Optional[str] = None,
    ):
        """
        Some attributes are protected and have to be set with setters:
          * set_contacts(...)
          * set_shank_ids(...)

        Parameters
        ----------
        ndim: 2 or 3, default: 2
            Handles 2D or 3D probe
        si_units: "um" | "mm" | "m", default: "um"
            The si units to use for the probe
        name: str | None, default: None
            The name of the probe
        serial_number: str | None, default: None
            The serial number of the probe
        model_name: str | None, default: None
            The model of the probe
        manufacturer: str | None, default: None
            The manufacturer of the probe

        Returns
        -------
        Probe: instance of Probe
        """

        assert ndim in (2, 3), "ndim can only be 2 or 3"
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
        self.annotations = dict()

        # set key properties
        self.name = name
        self.serial_number = serial_number
        self.model_name = model_name
        self.manufacturer = manufacturer

        # same idea but handle in vector way for contacts
        self.contact_annotations = dict()

        # the Probe can belong to a ProbeGroup
        self._probe_group = None

    @property
    def contact_positions(self):
        """The position of the center for each contact"""
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

    @property
    def name(self):
        return self.annotations.get("name", None)

    @name.setter
    def name(self, value):
        if value is not None:
            self.annotate(name=value)
        else:
            # we remove the annotation if it exists
            _ = self.annotations.pop("name", None)

    @property
    def serial_number(self):
        return self.annotations.get("serial_number", None)

    @serial_number.setter
    def serial_number(self, value):
        if value is not None:
            self.annotate(serial_number=value)
        else:
            # we remove the annotation if it exists
            _ = self.annotations.pop("serial_number", None)

    @property
    def model_name(self):
        return self.annotations.get("model_name", None)

    @model_name.setter
    def model_name(self, value):
        if value is not None:
            self.annotate(model_name=value)
        else:
            # we remove the annotation if it exists
            _ = self.annotations.pop("model_name", None)

    @property
    def manufacturer(self):
        return self.annotations.get("manufacturer", None)

    @manufacturer.setter
    def manufacturer(self, value):
        if value is not None:
            self.annotate(manufacturer=value)
        else:
            # we remove the annotation if it exists
            _ = self.annotations.pop("manufacturer", None)

    def get_title(self) -> str:
        if self.contact_positions is None:
            txt = "Undefined probe"
        else:
            n = self.get_contact_count()
            name = self.name
            serial_number = self.serial_number
            model_name = self.model_name
            manufacturer = self.manufacturer
            txt = ""
            if name is not None:
                txt += f"{name}"
            else:
                txt += f"Probe"
            if manufacturer is not None:
                txt += f" - {manufacturer}"
            if model_name is not None:
                txt += f" - {model_name}"
            if serial_number is not None:
                txt += f" - {serial_number}"
            txt += f" - {n}ch"
            if self.shank_ids is not None:
                num_shank = self.get_shank_count()
                txt += f" - {num_shank}shanks"
        return txt

    def __repr__(self):
        return self.get_title()

    def annotate(self, **kwargs):
        """
        Annotates the probe object.

        Parameters
        ----------
        **kwargs : list of keyword arguments to add to the annotations (e.g., brain_area="CA1")
        """
        self.annotations.update(kwargs)
        self.check_annotations()

    def annotate_contacts(self, **kwargs):
        """
        Annotates the contacts of the probe.

        Parameters
        ----------
        **kwargs : list of keyword arguments to add to the annotations (e.g., quality=["good", "bad", ...])
        """
        n = self.get_contact_count()
        for k, values in kwargs.items():
            assert len(values) == n, (
                f"annotate_contacts requires a list or array as values with length {n}, "
                f"you entered a value of type: {type(values)} and length of {len(values)}"
            )
            values = np.asarray(values)
            self.contact_annotations[k] = values

    def check_annotations(self):
        d = self.annotations
        if "first_index" in d:
            assert d["first_index"] in (0, 1), f"The 'first_index' must be 0 or 1, it is currently {d['first_index']}"

    def get_contact_count(self) -> int:
        """
        Return the number of contacts on the probe.
        """
        assert self.contact_positions is not None
        return len(self.contact_positions)

    def get_shank_count(self) -> int:
        """
        Return the number of shanks for this probe.
        """
        assert self.shank_ids is not None
        n = len(np.unique(self.shank_ids))
        return n

    def set_contacts(
        self, positions, shapes="circle", shape_params={"radius": 10}, plane_axes=None, contact_ids=None, shank_ids=None
    ):
        """Sets contacts to a Probe.

        This sets four attributes of the probe:
            contact_positions,
            contact_shapes,
            contact_shape_params,
            _contact_plane_axes

        Parameters
        ----------
        positions : array (num_contacts, ndim)
            Positions of contacts (2D or 3D depending on probe 'ndim').
        shapes : "circle" | "square" | "rect" | array, default: "circle"
            Shape of each contact ('circle'/'square'/'rect').
        shape_params : dict or list of dict, default: {"radius": 10}
            Contains kwargs for shapes:
            * "radius" for circle
            * "width" for square,
            * "width/height" for rect
        plane_axes : np.array (num_contacts, 2, ndim) | None, default: None
            Defines the two axes of the contact plane for each electrode.
            The third dimension corresponds to the probe `ndim` (2d or 3d).
        contact_ids: array[str] | None, default: None
            Defines the contact ids for the contacts. If None, contact ids are not assigned.
        shank_ids : array[str] | None, default: None
            Defines the shank ids for the contacts. If None, then
            these are assigned to a unique Shank.
        """
        positions = np.array(positions)
        if positions.shape[1] != self.ndim:
            raise ValueError(f"positions.shape[1]: {positions.shape[1]} and ndim: {self.ndim} do not match!")

        self._contact_positions = positions
        n = positions.shape[0]

        # This defines the contact plane (2D or 3D) along which the contacts lie.
        # For 2D we make auto
        if plane_axes is None:
            if self.ndim == 3:
                raise ValueError("For ndim==3, you need to give a 'plane_axes'")
            else:
                plane_axes = np.zeros((n, 2, self.ndim))
                plane_axes[:, 0, 0] = 1
                plane_axes[:, 1, 1] = 1
        plane_axes = np.array(plane_axes)
        self._contact_plane_axes = plane_axes

        if contact_ids is not None:
            self.set_contact_ids(contact_ids)

        if shank_ids is None:
            self._shank_ids = np.zeros(n, dtype=str)
        else:
            self._shank_ids = np.asarray(shank_ids).astype(str)
            if self.shank_ids.size != n:
                raise ValueError(f"shank_ids have wrong size: {self.shanks.ids.size} != {n}")

        # shape
        if isinstance(shapes, str):
            shapes = [shapes] * n
        shapes = np.array(shapes)
        if not np.all(np.isin(shapes, _possible_contact_shapes)):
            raise ValueError(f"contacts shape must be in {_possible_contact_shapes}")
        if shapes.shape[0] != n:
            raise ValueError(f"contacts shape {shapes.shape[0]} must have same length as positions {n}")
        self._contact_shapes = np.array(shapes)

        # shape params
        if isinstance(shape_params, dict):
            shape_params = [shape_params] * n
        self._contact_shape_params = np.array(shape_params)

    def set_planar_contour(self, contour_polygon: list):
        """Set the planar contour (the shape) of the probe.

        Parameters
        ----------
        contour_polygon : list
            List of contour points (2D or 3D depending on ndim)
        """
        contour_polygon = np.asarray(contour_polygon)
        if contour_polygon.shape[1] != self.ndim:
            raise ValueError(f"contour_polygon.shape[1] {contour_polygon.shape[1]} and ndim {self.ndim} do not match!")
        self.probe_planar_contour = contour_polygon

    def create_auto_shape(self, probe_type: "tip" | "rect" | "circular" = "tip", margin: float = 20.0):
        """Create a planar contour automatically based on probe contact positions.

        This function generates a 2D polygon that outlines the shape of the probe, adjusted
        by a specified margin. The resulting contour is set as the planar contour of the probe.

        Parameters
        ----------
        probe_type : {"tip", "rect", "circular"}, default: "tip"
            The type of probe used to collect contact data:

            * "tip": Assumes a single-point contact probe. The generated contour is
            a rectangle with a triangular "tip" extending downwards.
            * "rect": Assumes a rectangular contact probe. The generated contour is
            a rectangle.
            * "circular": Assumes a circular contact probe. The generated contour
            is a circle.

        margin : float, default: 20.0
            The margin to add around the contact positions. The behavior varies by
            probe type:

            * "tip": The margin is added around the rectangular portion of the contour
            and to the base of the tip. The tip itself is extended downwards by
            four times the margin value.
            * "rect": The margin is added evenly around all sides of the rectangle.
            * "circular": The margin is added to the radius of the circle.

        Notes
        -----
        This function is designed for 2D data only. If you have 3D data, consider projecting
        it onto a plane before using this method.
        """
        if self.ndim != 2:
            raise ValueError(f"Auto shape is supported only for 2d, you have ndim {self.ndim}")

        if self._shank_ids is None:
            shank_ids = np.zeros((self.get_contact_count()), dtype="int64")
        else:
            shank_ids = self._shank_ids

        polygon = []

        for i, shank_id in enumerate(np.unique(shank_ids)):
            mask = shank_ids == shank_id

            x0 = np.min(self.contact_positions[mask, 0])
            x1 = np.max(self.contact_positions[mask, 0])
            x0 -= margin
            x1 += margin

            y0 = np.min(self.contact_positions[:, 1])
            y1 = np.max(self.contact_positions[:, 1])
            y0 -= margin
            y1 += margin

            if probe_type == "rect":
                polygon += [
                    (x0, y1),
                    (x0, y0),
                    (x1, y0),
                    (x1, y1),
                ]
            elif probe_type == "tip":
                tip = ((x0 + x1) * 0.5, y0 - margin * 4)
                polygon += [
                    (x0, y1),
                    (x0, y0),
                    tip,
                    (x1, y0),
                    (x1, y1),
                ]
            elif probe_type == "circular":
                radius_x = (x1 - x0) / 2
                radius_y = (y1 - y0) / 2
                center = ((x0 + x1) / 2, (y0 + y1) / 2)
                radius = max(radius_x, radius_y) + margin
                num_vertices = 100
                theta = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                vertices = np.vstack((x, y)).T
                polygon += vertices.tolist()
            else:
                raise ValueError(f"'probe_type' can only be 'rect, 'tip' or 'circular', you have entered {probe_type}")

        self.set_planar_contour(polygon)

    def set_device_channel_indices(self, channel_indices: np.array | list):
        """
        Manually set the device channel indices.

        If some channels are not connected or not recorded then channel should be set to "-1"

        Parameters
        ----------
        channel_indices : array[int] | list[int]
            The device channel indices to set
        """
        channel_indices = np.asarray(channel_indices, dtype=int)
        if channel_indices.size != self.get_contact_count():
            ValueError(
                f"channel_indices {channel_indices.size} do not have "
                f"the same size as contacts {self.get_contact_count()}"
            )
        self.device_channel_indices = channel_indices
        if self._probe_group is not None:
            self._probe_group.check_global_device_wiring_and_ids()

    def wiring_to_device(self, pathway: str, channel_offset: int = 0):
        """
        Automatically set device_channel_indices based on a pathway.

        See probeinterface.get_available_pathways()

        Parameters
        ----------
        pathway : str
           The pathway. E.g. 'H32>RHD'
        channel_offset: int, default: 0
            An optional offset to add to the device_channel_indices
        """
        from .wiring import wire_probe

        wire_probe(self, pathway, channel_offset=channel_offset)

    def set_contact_ids(self, contact_ids: np.array | list):
        """
        Set contact ids. Channel ids are converted to strings.
        Contact ids must be **unique** for the **Probe**
        and also for the **ProbeGroup**

        Parameters
        ----------
        contact_ids : list or array
            Array with contact ids. If contact_ids are int or float they are converted to str

        """
        contact_ids = np.asarray(contact_ids)
        if np.all([c == "" for c in contact_ids]):
            self._contact_ids = None
            return

        assert np.unique(contact_ids).size == contact_ids.size, "Contact ids have to be unique within a Probe"

        if contact_ids.size != self.get_contact_count():
            ValueError(
                f"contact_ids {contact_ids.size} do not have the same size "
                f"as number of contacts {self.get_contact_count()}"
            )

        if contact_ids.dtype.kind != "U":
            contact_ids = contact_ids.astype("U")

        self._contact_ids = contact_ids
        if self._probe_group is not None:
            self._probe_group.check_global_device_wiring_and_ids()

    def set_shank_ids(self, shank_ids: np.array | list):
        """
        Set shank ids.

        Parameters
        ----------
        shank_ids : list or array
            Array with shank ids, if int or float converted to strings
        """
        shank_ids = np.asarray(shank_ids).astype(str)
        if shank_ids.size != self.get_contact_count():
            raise ValueError(
                f"shank_ids have wrong size. Has to match number " f"of contacts: {self.get_contact_count()}"
            )
        self._shank_ids = shank_ids

    def get_shanks(self):
        """
        Return the list of Shank objects for this Probe
        """
        assert self.shank_ids is not None, "Can only get shanks if `shank_ids` exist"
        shanks = []
        for shank_id in np.unique(self.shank_ids):
            shank = Shank(probe=self, shank_id=shank_id)
            shanks.append(shank)
        return shanks

    def __eq__(self, other):
        if not isinstance(other, Probe):
            return False

        if not (
            self.ndim == other.ndim
            and self.si_units == other.si_units
            and self.name == other.name
            and self.serial_number == other.serial_number
            and self.model_name == other.model_name
            and self.manufacturer == other.manufacturer
            and np.array_equal(self._contact_positions, other._contact_positions)
            and np.array_equal(self._contact_plane_axes, other._contact_plane_axes)
            and np.array_equal(self._contact_shapes, other._contact_shapes)
            and np.array_equal(self._contact_shape_params, other._contact_shape_params)
            and np.array_equal(self.probe_planar_contour, other.probe_planar_contour)
            and np.array_equal(self._shank_ids, other._shank_ids)
            and np.array_equal(self.device_channel_indices, other.device_channel_indices)
            and np.array_equal(self._contact_ids, other._contact_ids)
            and self.annotations == other.annotations
        ):
            return False

        # Compare contact_annotations dictionaries
        if self.contact_annotations.keys() != other.contact_annotations.keys():
            return False
        for key in self.contact_annotations:
            if not np.array_equal(self.contact_annotations[key], other.contact_annotations[key]):
                return False

        # planar contour
        if self.probe_planar_contour is not None:
            if other.probe_planar_contour is None:
                return False
            if not np.array_equal(self.probe_planar_contour, other.probe_planar_contour):
                return False

        return True

    def copy(self):
        """
        Copy to another Probe instance.

        Note: device_channel_indices are not copied
        and contact_ids are not copied
        """
        other = Probe()
        other.set_contacts(
            positions=self.contact_positions.copy(),
            plane_axes=self.contact_plane_axes.copy(),
            shapes=self.contact_shapes.copy(),
            shape_params=self.contact_shape_params.copy(),
        )
        if self.probe_planar_contour is not None:
            other.set_planar_contour(self.probe_planar_contour.copy())
        # channel_indices are not copied
        return other

    def to_3d(self, axes: "xy" | "yz" | "xz" = "xz"):
        """
        Transform 2d probe to 3d probe.

        Note: device_channel_indices are not copied.

        Parameters
        ----------
        axes : "xy" | "yz" | "xz", default: "xz"
            The axes that define the plane on which the 2D probe is defined. 'xy', 'yz' ', xz'
        """
        assert self.ndim == 2, "to convert to_3d you should start with a 2d probe"

        probe3d = Probe(ndim=3, si_units=self.si_units)

        # contacts
        positions = _2d_to_3d(self.contact_positions, axes)
        plane0 = _2d_to_3d(self.contact_plane_axes[:, 0, :], axes)
        plane1 = _2d_to_3d(self.contact_plane_axes[:, 1, :], axes)
        plane_axes = np.concatenate([plane0[:, np.newaxis, :], plane1[:, np.newaxis, :]], axis=1)
        probe3d.set_contacts(
            positions=positions,
            plane_axes=plane_axes,
            shapes=self.contact_shapes.copy(),
            shape_params=self.contact_shape_params.copy(),
        )

        # shape
        if self.probe_planar_contour is not None:
            vertices3d = _2d_to_3d(self.probe_planar_contour, axes)
            probe3d.set_planar_contour(vertices3d)

        if self.device_channel_indices is not None:
            probe3d.device_channel_indices = self.device_channel_indices

        return probe3d

    def to_2d(self, axes: "xy" | "yz" | "xz" = "xy"):
        """
        Transform 3d probe to 2d probe.

        Note: device_channel_indices are not copied.

        Parameters
        ----------
        plane : "xy" | "yz" | "xz", default: "xy"
            The plane on which the 2D probe is defined.
        """
        assert self.ndim == 3, "To use to_2d you should start with a 3d probe"

        probe2d = Probe(ndim=2, si_units=self.si_units)

        # contacts
        positions = _3d_to_2d(self.contact_positions, axes)
        probe2d.set_contacts(
            positions=positions, shapes=self.contact_shapes.copy(), shape_params=self.contact_shape_params.copy()
        )

        # shape
        if self.probe_planar_contour is not None:
            vertices3d = _3d_to_2d(self.probe_planar_contour, axes)
            probe2d.set_planar_contour(vertices3d)

        if self.device_channel_indices is not None:
            probe2d.device_channel_indices = self.device_channel_indices

        return probe2d

    def get_contact_vertices(self) -> list:
        """
        Return a list of contact vertices.
        """

        vertices = []
        for i in range(self.get_contact_count()):
            shape = self.contact_shapes[i]
            shape_param = self.contact_shape_params[i]
            plane_axe = self.contact_plane_axes[i]
            pos = self.contact_positions[i]
            if shape == "circle":
                r = shape_param["radius"]
                theta = np.linspace(0, 2 * np.pi, 360)
                theta = np.tile(theta[:, np.newaxis], [1, self.ndim])
                one_vertice = pos + r * np.cos(theta) * plane_axe[0] + r * np.sin(theta) * plane_axe[1]
            elif shape == "square":
                w = shape_param["width"]
                one_vertice = [
                    pos - w / 2 * plane_axe[0] - w / 2 * plane_axe[1],
                    pos - w / 2 * plane_axe[0] + w / 2 * plane_axe[1],
                    pos + w / 2 * plane_axe[0] + w / 2 * plane_axe[1],
                    pos + w / 2 * plane_axe[0] - w / 2 * plane_axe[1],
                ]
                one_vertice = np.array(one_vertice)
            elif shape == "rect":
                w = shape_param["width"]
                h = shape_param["height"]
                one_vertice = [
                    pos - w / 2 * plane_axe[0] - h / 2 * plane_axe[1],
                    pos - w / 2 * plane_axe[0] + h / 2 * plane_axe[1],
                    pos + w / 2 * plane_axe[0] + h / 2 * plane_axe[1],
                    pos + w / 2 * plane_axe[0] - h / 2 * plane_axe[1],
                ]
                one_vertice = np.array(one_vertice)
            else:
                raise ValueError(f"'shape' of {shape} is not supported")
            vertices.append(one_vertice)
        return vertices

    def move(self, translation_vector: np.array | list):
        """
        Translate the probe in one direction.

        Parameters
        ----------
        translation_vector : list or array
            The translation vector in shape 2D or 3D
        """

        translation_vector = np.asarray(translation_vector)
        assert translation_vector.shape[0] == self.ndim

        self._contact_positions += translation_vector

        if self.probe_planar_contour is not None:
            self.probe_planar_contour += translation_vector

    def rotate(self, theta: float, center: list | np.ndarray | None = None, axis: "xy" | "yz" | "xz" | None = None):
        """
        Rotate the probe around a specified axis.

        Parameters
        ----------
        theta : float
            In degrees, anticlockwise/counterclockwise
        center : array | list |  None, default: None
            Center of rotation. If None, the center of probe is used
        axis : "xy" | "yz" | "xz" | None, default: None
            Axis of rotation.
            It must be None for 2D probes
            It must be given for 3D probes

        """

        if center is None:
            center = np.mean(self.contact_positions, axis=0)

        center = np.asarray(center)
        assert center.size == self.ndim, f"If center given it must have size of ndim: {center.size} != {self.ndim}"
        center = center[None, :]

        theta = np.deg2rad(theta)

        if self.ndim == 2:
            assert axis is None, "axis must be None for 2d probes"
            R = _rotation_matrix_2d(theta)
        elif self.ndim == 3:
            assert axis is not None, "axis must be specified for 3d probes"
            R = _rotation_matrix_3d(axis, theta).T

        new_positions = (self.contact_positions - center) @ R + center

        new_plane_axes = np.zeros_like(self.contact_plane_axes)
        for i in range(2):
            new_plane_axes[:, i, :] = (
                (self.contact_plane_axes[:, i, :] - center + self.contact_positions) @ R + center - new_positions
            )

        self._contact_positions = new_positions
        self._contact_plane_axes = new_plane_axes

        if self.probe_planar_contour is not None:
            new_vertices = (self.probe_planar_contour - center) @ R + center
            self.probe_planar_contour = new_vertices

    def rotate_contacts(self, thetas: float | np.array[float] | list[float]):
        """
        Rotate each contact of the probe.
        Internally, it modifies the contact_plane_axes.

        Parameters
        ----------
        thetas : float | array[float] | list[float]
            Rotation angle in degrees.
            If scalar, then it is applied to all contacts.

        """

        if self.ndim == 3:
            raise ValueError("By contact rotation is implemented only for 2d")

        n = self.get_contact_count()

        if isinstance(thetas, (int, float)):
            thetas = np.array([thetas] * n, dtype="float64")

        thetas = np.deg2rad(thetas)

        for e in range(n):
            R = _rotation_matrix_2d(thetas[e])
            for i in range(2):
                self.contact_plane_axes[e, i, :] = self.contact_plane_axes[e, i, :] @ R

    _dump_attr_names = [
        "ndim",
        "si_units",
        "annotations",
        "contact_annotations",
        "_contact_positions",
        "_contact_plane_axes",
        "_contact_shapes",
        "_contact_shape_params",
        "probe_planar_contour",
        "device_channel_indices",
        "_contact_ids",
        "_shank_ids",
    ]

    def to_dict(self, array_as_list: bool = False) -> dict:
        """Create a dictionary of all necessary attributes.
        Useful for dumping and saving to json.

        Parameters
        ----------
        array_as_list : bool, default: False
            If True, arrays are converted to lists

        Returns
        -------
        d : dict
            The dictionary representation of the probe
        """
        d = {}
        for k in self._dump_attr_names:
            v = getattr(self, k, None)
            if array_as_list and isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        v[kk] = vv.tolist()
            if array_as_list and v is not None and isinstance(v, np.ndarray):
                v = v.tolist()
            if v is not None:
                if k.startswith("_"):
                    d[k[1:]] = v
                else:
                    d[k] = v
        return d

    @staticmethod
    def from_dict(d: dict) -> "Probe":
        """Instantiate a Probe from a dictionary

        Parameters
        ----------
        d : dict
            The dictionary representation of the probe

        Returns
        -------
        probe : Probe
            The instantiated Probe object
        """
        probe = Probe(ndim=d["ndim"], si_units=d["si_units"])

        probe.set_contacts(
            positions=d["contact_positions"],
            plane_axes=d["contact_plane_axes"],
            shapes=d["contact_shapes"],
            shape_params=d["contact_shape_params"],
        )

        v = d.get("probe_planar_contour", None)
        if v is not None:
            probe.set_planar_contour(v)

        v = d.get("device_channel_indices", None)
        if v is not None:
            probe.set_device_channel_indices(v)

        v = d.get("shank_ids", None)
        if v is not None:
            probe.set_shank_ids(v)

        v = d.get("contact_ids", None)
        if v is not None:
            probe.set_contact_ids(v)

        if "annotations" in d:
            probe.annotate(**d["annotations"])
        if "contact_annotations" in d:
            probe.annotate_contacts(**d["contact_annotations"])

        return probe

    def to_numpy(self, complete: bool = False) -> np.array:
        """
        Export the probe to a numpy structured array.
        This array handles all contact attributes.

        Similar to the 'to_dataframe()' pandas function, but without pandas dependency.

        The intended use is to attach this array to a recording object as a property ("contact vector")

        Parameters
        ----------
        complete : bool, default: False
            If True, export complete information about the probe,
            including contact_plane_axes/si_units/device_channel_indices.

        Returns
        -------
        arr : numpy.array
            Structured array with the following dtype schema:

            dtype = [
                ('x', 'float64'),
                ('y', 'float64'),
                ('z', 'float64', optional),
                ('contact_shapes', 'U64'),
                # Shape parameters
                ('shape_param_1', 'float64'),
                ('shape_param_2', 'float64'),
                                ⋮
                                ⋮
                                ⋮
                variable number of shape parameters
                ...
                ('shank_ids', 'U64'),
                ('contact_ids', 'U64'),

                # The rest is added only if `complete=True`
                ('device_channel_indices', 'int64', optional),
                ('si_units', 'U64', optional),
                ('plane_axis_x_0', 'float64', optional),
                ('plane_axis_x_1', 'float64', optional),
                ('plane_axis_y_0', 'float64', optional),
                ('plane_axis_y_1', 'float64', optional),
                ('plane_axis_z_0', 'float64', optional),
                ('plane_axis_z_1', 'float64', optional),
                # Annotations
                ('annotation_name_1', 'dtype of annotation', optional),
                ('annotation_name_2', 'dtype of annotation', optional),                                ⋮
                                ⋮
                                ⋮
                                ⋮
                variable number of annotations
                ...
            ]
        """

        # First define the dtype
        dtype = [("x", "float64"), ("y", "float64")]
        if self.ndim == 3:
            dtype += [("z", "float64")]

        dtype += [("contact_shapes", "U64")]
        param_shape = []
        for i, p in enumerate(self.contact_shape_params):
            for k, v in p.items():
                if k not in param_shape:
                    param_shape.append(k)
        for k in param_shape:
            dtype += [(k, "float64")]
        dtype += [("shank_ids", "U64"), ("contact_ids", "U64")]

        if complete:
            dtype += [("device_channel_indices", "int64")]
            dtype += [("si_units", "U64")]
            for i in range(self.ndim):
                dim = ["x", "y", "z"][i]
                dtype += [(f"plane_axis_{dim}_0", "float64")]
                dtype += [(f"plane_axis_{dim}_1", "float64")]
            for annotation_name, annotation_values in self.contact_annotations.items():
                dtype += [(f"{annotation_name}", np.asarray(annotation_values).dtype)]

        # Then add the data to the structured array
        arr = np.zeros(self.get_contact_count(), dtype=dtype)
        arr["x"] = self.contact_positions[:, 0]
        arr["y"] = self.contact_positions[:, 1]
        if self.ndim == 3:
            arr["z"] = self.contact_positions[:, 2]
        arr["contact_shapes"] = self.contact_shapes
        for i, p in enumerate(self.contact_shape_params):
            for k, v in p.items():
                arr[k][i] = v

        arr["shank_ids"] = self.shank_ids

        if self.contact_ids is None:
            arr["contact_ids"] = [""] * self.get_contact_count()
        else:
            arr["contact_ids"] = self.contact_ids

        if complete:
            arr["si_units"] = self.si_units

            # (num_contacts, 2, ndim)
            for i in range(self.ndim):
                dim = ["x", "y", "z"][i]
                arr[f"plane_axis_{dim}_0"] = self.contact_plane_axes[:, 0, i]
                arr[f"plane_axis_{dim}_1"] = self.contact_plane_axes[:, 1, i]

            if self.device_channel_indices is None:
                arr["device_channel_indices"] = -1
            else:
                arr["device_channel_indices"] = self.device_channel_indices

            for annotation_name, annotation_values in self.contact_annotations.items():
                arr[annotation_name] = annotation_values

        return arr

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Probe":
        """
        Create Probe from a complex numpy array
        see Probe.to_numpy()

        Parameters
        ----------
        arr : np.array
            The structured np.array representation of the probe

        Returns
        -------
        probe : Probe
            The instantiated Probe object
        """
        fields = list(arr.dtype.fields)
        main_fields = [
            "x",
            "y",
            "z",
            "contact_shapes",
            "shank_ids",
            "contact_ids",
            "device_channel_indices",
            "radius",
            "width",
            "height",
            "plane_axis_x_0",
            "plane_axis_x_1",
            "plane_axis_y_0",
            "plane_axis_y_1",
            "plane_axis_z_0",
            "plane_axis_z_1",
            "probe_index",
            "si_units",
        ]
        contact_annotation_fields = [f for f in fields if f not in main_fields]

        if "z" in fields:
            ndim = 3
        else:
            ndim = 2

        assert "x" in fields, "arr must contain a .dtype.fields of x"
        assert "y" in fields, "arr must contain a .dtype.fields of y"
        if "si_units" in fields:
            assert np.unique(arr["si_units"]).size == 1
            si_units = np.unique(arr["si_units"])[0]
        else:
            si_units = "um"
        probe = Probe(ndim=ndim, si_units=si_units)

        # contacts
        positions = np.zeros((arr.size, ndim), dtype="float64")
        for i, dim in enumerate(["x", "y", "z"][:ndim]):
            positions[:, i] = arr[dim]
        shapes = arr["contact_shapes"]
        shape_params = []
        for i, shape in enumerate(shapes):
            if shape == "circle":
                p = {"radius": float(arr["radius"][i])}
            elif shape == "square":
                p = {"width": float(arr["width"][i])}
            elif shape == "rect":
                p = {"width": float(arr["width"][i]), "height": float(arr["height"][i])}
            else:
                raise ValueError("You are in bad shape")
            shape_params.append(p)

        if "plane_axis_x_0" in fields:
            # (num_contacts, 2, ndim)
            plane_axes = np.zeros((arr.size, 2, ndim))
            for i in range(ndim):
                dim = ["x", "y", "z"][i]
                plane_axes[:, 0, i] = arr[f"plane_axis_{dim}_0"]
                plane_axes[:, 1, i] = arr[f"plane_axis_{dim}_1"]
        else:
            plane_axes = None

        probe.set_contacts(positions=positions, plane_axes=plane_axes, shapes=shapes, shape_params=shape_params)

        if "device_channel_indices" in fields:
            dev_channel_indices = arr["device_channel_indices"]
            if not np.all(dev_channel_indices == -1):
                probe.set_device_channel_indices(dev_channel_indices)
        if "shank_ids" in fields:
            probe.set_shank_ids(arr["shank_ids"])
        if "contact_ids" in fields:
            probe.set_contact_ids(arr["contact_ids"])

        # contact annotations
        for k in contact_annotation_fields:
            probe.annotate_contacts(**{k: arr[k]})
        return probe

    def add_probe_to_zarr_group(self, group: "zarr.Group") -> None:
        """
        Serialize the probe's data and structure to a specified Zarr group.

        This method is used to save the probe's attributes, annotations, and other
        related data into a Zarr group, facilitating integration into larger Zarr
        structures.

        Parameters
        ----------
        group : zarr.Group
            The target Zarr group where the probe's data will be stored.
        """
        probe_arr = self.to_numpy(complete=True)

        # add fields and contact annotations
        for field_name, (dtype, offset) in probe_arr.dtype.fields.items():
            data = probe_arr[field_name]
            group.create_dataset(name=field_name, data=data, dtype=dtype, chunks=False)

        # Annotations as a group (special attibutes are stored as annotations)
        annotations_group = group.create_group("annotations")
        for key, value in self.annotations.items():
            annotations_group.attrs[key] = value

        # Add planar contour
        if self.probe_planar_contour is not None:
            group.create_dataset(
                name="probe_planar_contour", data=self.probe_planar_contour, dtype="float64", chunks=False
            )

    def to_zarr(self, folder_path: str | Path) -> None:
        """
        Serialize the Probe object to a Zarr file located at the specified folder path.

        This method initializes a new Zarr group at the given folder path and calls
        `add_probe_to_zarr_group` to serialize the Probe's data into this group, effectively
        storing the entire Probe's state in a Zarr archive.

        Parameters
        ----------
        folder_path : str | Path
            The path to the folder where the Zarr data structure will be created and
            where the serialized data will be stored. If the folder does not exist,
            it will be created.
        """
        import zarr

        # Create or open a Zarr group for writing
        zarr_group = zarr.open_group(folder_path, mode="w")

        # Serialize this Probe object into the Zarr group
        self.add_probe_to_zarr_group(zarr_group)

    @staticmethod
    def from_zarr_group(group: "zarr.Group") -> "Probe":
        """
        Load a probe instance from a given Zarr group.

        Parameters
        ----------
        group : zarr.Group
            The Zarr group from which to load the probe.

        Returns
        -------
        Probe
            An instance of the Probe class initialized with data from the Zarr group.
        """
        import zarr

        dtype = []
        # load all datasets
        num_contacts = None
        probe_arr_keys = []
        for key in group.keys():
            if key == "probe_planar_contour":
                continue
            if key == "annotations":
                continue
            dset = group[key]
            if isinstance(dset, zarr.Array):
                probe_arr_keys.append(key)
                dtype.append((key, dset.dtype))
                if num_contacts is None:
                    num_contacts = len(dset)

        # Create a structured array from the datasets
        probe_arr = np.zeros(num_contacts, dtype=dtype)

        for probe_key in probe_arr_keys:
            probe_arr[probe_key] = group[probe_key][:]

        # Create a Probe instance from the structured array
        probe = Probe.from_numpy(probe_arr)

        # Load annotations
        annotations_group = group.get("annotations", None)
        for key in annotations_group.attrs.keys():
            # Use the annotate method for each key-value pair
            probe.annotate(**{key: annotations_group.attrs[key]})

        if "probe_planar_contour" in group:
            # Directly assign since there's no specific setter for probe_planar_contour
            probe.probe_planar_contour = group["probe_planar_contour"][:]

        return probe

    @staticmethod
    def from_zarr(folder_path: str | Path) -> "Probe":
        """
        Deserialize the Probe object from a Zarr file located at the given folder path.

        Parameters
        ----------
        folder_path : str | Path
            The path to the folder where the Zarr file is located.

        Returns
        -------
        Probe
            An instance of the Probe class initialized with data from the Zarr file.
        """
        import zarr

        zarr_group = zarr.open(folder_path, mode="r")
        return Probe.from_zarr_group(zarr_group)

    def to_dataframe(self, complete: bool = False) -> "pandas.DataFrame":
        """
        Export the probe to a pandas dataframe

        Parameters
        ----------
        complete : bool, default: False
            If True, export complete information about the probe,
            including the probe plane axis.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe representation of the probe

        """

        import pandas as pd

        arr = self.to_numpy(complete=complete)
        df = pd.DataFrame.from_records(arr)
        df.index = np.arange(df.shape[0], dtype="int64")
        return df

    @staticmethod
    def from_dataframe(df: "pandas.DataFrame") -> "Probe":
        """
        Create Probe from a pandas.DataFrame
        see Probe.to_dataframe()

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe representation of the probe

        Returns
        -------
        probe : Probe
            The instantiated Probe object

        """
        arr = df.to_records(index=False)
        return Probe.from_numpy(arr)

    def to_image(
        self,
        values: np.array | list,
        pixel_size: float = 0.5,
        num_pixel: Optional[int] = None,
        method: "linear" | "nearest" | "cubic" = "linear",
        xlims: Optional[tuple] = None,
        ylims: Optional[tuple] = None,
    ) -> tuple[np.ndarray, tuple, tuple]:
        """
        Generated a 2d (image) from a values vector with an interpolation
        into a grid mesh.

        Parameters
        ----------
        values : np.ndarray | list
            vector same size as contact number to be color plotted
        pixel_size : float, default: 0.5
            size of one pixel in micrometers
        num_pixel : Optional[int] | None, default: None
            alternative to pixel_size give pixel number of the image width
        method : "linear" | "nearest" | "cubic", default: "linear"
            Method of interpolation to generate a grid mesh
        xlims : Optional[tuple], default: None
            Force image xlims
        ylims : Optional[tuple], default: None
            Force image ylims

        Returns
        -------
        image : 2d array
            The generated image
        xlims : tuple
            The x limits
        ylims : tuple
            The y limits

        """
        try:
            from scipy.interpolate import griddata
        except ImportError:
            raise ImportError("to_image() requires the scipy package")
        assert self.ndim == 2, "only 2d probes can be used in to_image"
        assert values.shape == (self.get_contact_count(),), (
            f"Shape mismatch: values {values.shape} must have the "
            f"same size as contact count {self.get_contact_count()}"
        )

        if xlims is None:
            x0 = np.min(self.contact_positions[:, 0])
            x1 = np.max(self.contact_positions[:, 0])
            xlims = (x0, x1)

        if ylims is None:
            y0 = np.min(self.contact_positions[:, 1])
            y1 = np.max(self.contact_positions[:, 1])
            ylims = (y0, y1)

        x0, x1 = xlims
        y0, y1 = ylims

        if num_pixel is not None:
            pixel_size = (x1 - x0) / num_pixel

        grid_x, grid_y = np.meshgrid(np.arange(x0, x1, pixel_size), np.arange(y0, y1, pixel_size))
        image = griddata(self.contact_positions, values, (grid_x, grid_y), method=method)

        if method == "nearest":
            # hack to force nan when nereast to avoid interpolation in the full rectangle
            image2, _, _ = self.to_image(values, pixel_size=pixel_size, method="linear", xlims=xlims, ylims=ylims)
            image[np.isnan(image2)] = np.nan

        return image, xlims, ylims

    def get_slice(self, selection: np.ndarray[bool | int]):
        """
        Get a copy of the Probe with a sub selection of contacts.

        Selection can be boolean or by index

        Parameters
        ----------
        selection : np.array of bool or int (for index)
            Either an np.array of bool or for desired selection of contacts
            or the indices of the desired contacts

        Returns
        -------
        sliced_probe: Probe
            The sliced probe

        """

        n = self.get_contact_count()

        selection = np.asarray(selection)
        if selection.dtype == "bool":
            assert selection.shape == (n,), (
                f"if array of bool given it must be the same size " "as the number of contacts {selection.shape} != {n}"
            )
        elif selection.dtype.kind == "i":
            assert np.unique(selection).size == selection.size
            assert 0 <= np.min(selection) < n, f"An index within your selection is out of bounds {np.min(selection)}"
            assert 0 <= np.max(selection) < n, f"An index within your selection is out of bounds {np.max(selection)}"
        else:
            raise TypeError(f"selection must be bool array or int array, not of type: {type(selection)}")

        d = self.to_dict(array_as_list=False)
        for k, v in d.items():
            if k == "probe_planar_contour":
                continue

            if isinstance(v, np.ndarray):
                assert v.shape[0] == n
                d[k] = v[selection].copy()

            if k == "contact_annotations":
                d[k] = {}
                for kk, vv in v.items():
                    d[k][kk] = vv[selection].copy()

        sliced_probe = Probe.from_dict(d)

        return sliced_probe


def _2d_to_3d(data2d: np.ndarray, axes: "xy" | "yz" | "xz") -> np.ndarray:
    """
    Add a third dimension on the given axes

    Parameters
    ----------
    data2d: np.array
        shape (n, 2)
    axes: "xy" | "yz"| "xz"
        The axes that define the plane where electrodes lie on.

    Returns
    -------
    data3d: np.ndarray
        shape (n, 3)

    """
    data3d = np.zeros((data2d.shape[0], 3), dtype=data2d.dtype)
    dims = np.array(["xyz".index(axis) for axis in axes])
    assert len(axes) == 2, "_2d_to_3d: axes should contain 2 dimensions!"
    data3d[:, dims] = data2d
    return data3d


def select_axes(data: np.ndarray, axes: "xy" | "yz" | "xz" = "xy") -> np.ndarray:
    """
    Select axes in a 3d or 2d array.

    Parameters
    ----------
    data: np.array
        shape (n, 2) or (n, 3)
    axes: "xy" | "yz" | "xz" | "xyz", default: "xy"
        The axis of selection

    Returns
    -------
    data3d
        shape (n, 3)

    """
    assert np.all([axes.count(axis) == 1 for axis in axes]), "select_axes : axes must be unique."
    dims = np.array(["xyz".index(axis) for axis in axes])
    assert data.shape[1] >= max(dims), f"Inconsistent shapes between positions {data.shape[1]} and axes {max(dims)}"
    return data[:, dims]


def _3d_to_2d(data3d: np.ndarray, axes: "xy" | "yz" | "xz" = "xy") -> np.ndarray:
    """
    Reduce 3d array to 2d array on given axes.

    Parameters
    ----------
    data: np.ndarray
        The data with shape (n,3)
    axes: "xy" | "yz" | "xz" default: "xy"
        The axes over which to reduce the 2d array

    Returns
    -------
    reduced_data: np.ndarray
        The reduced data array

    """
    assert data3d.shape[1] == 3, "To use _3d_to_2d should start with 3d data"
    assert len(axes) == 2, "axes should be one of 'xy' 'yz' or 'xz'"
    return select_axes(data3d, axes=axes)


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Returns 2D rotation matrix

    Parameters
    ----------
    theta : float
        Angle in radians for rotation (anti-clockwise/counterclockwise)

    Returns
    -------
    R : np.array
        2D rotation matrix

    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R


def _rotation_matrix_3d(axis: np.array | list, theta: float) -> np.ndarray:
    """
    Returns 3D rotation matrix

    Copy/paste from MEAutility

    Parameters
    ----------
    axis : np.array or list
        3D axis of rotation
    theta : float
        Angle in radians for rotation anti-clockwise/counterclockwise

    Returns
    -------
    R : np.array
        3D rotation matrix

    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    return R
