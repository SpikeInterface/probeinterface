from copy import deepcopy
import numpy as np
from .utils import generate_unique_ids
from .probe import Probe


class ProbeGroup:
    """
    Class to handle a group of Probe objects and the global wiring to a device.

    Optionally, it can handle the location of different probes.

    """

    def __init__(self):
        self._contact_array = None
        self._num_probes = 0
        self._probe_contours = []
        self._annotations = []

    @property
    def num_probes(self) -> int:
        """
        Get the number of probes in the ProbeGroup

        Returns
        -------
        num_probes: int
            The number of probes in the ProbeGroup
        """
        return int(self._num_probes)

    @property
    def probes(self) -> list[Probe]:
        """
        Get the list of probes in the ProbeGroup

        Returns
        -------
        probes: list of Probe
            The list of probes in the ProbeGroup
        """
        probes = []
        for probe_index in range(self._num_probes):
            probe_mask = self._contact_array["probe_index"] == probe_index
            probe_array = self._contact_array[probe_mask]
            probe = Probe.from_numpy(probe_array)
            # add annotations and contour
            probe.annotations = self._annotations[probe_index]
            probe.probe_planar_contour = self._probe_contours[probe_index]
            probe._probe_group = self
            probes.append(probe)
        return probes

    def annotate_probe(self, probe_index: int, **annotations) -> None:
        """
        Add annotations to a specific probe in the ProbeGroup

        Parameters
        ----------
        probe_index: int
            The index of the probe to annotate
        **annotations:
            The annotations to add to the probe

        """
        if probe_index >= self._num_probes:
            raise ValueError(f"probe_index {probe_index} is out of bounds for num_probes {self._num_probes}")
        self._annotations[probe_index].update(annotations)

    def add_probe(self, probe: Probe) -> None:
        """
        Add an additional probe to the ProbeGroup

        Parameters
        ----------
        probe: Probe
            The probe to add to the ProbeGroup

        """
        if len(self.probes) > 0:
            self._check_compatible(probe)

        probe_array = probe.to_numpy(complete=True, probe_index=self._num_probes)
        probe_dtype = probe_array.dtype
        if probe.contact_ids is None:
            count = probe.get_contact_count()
            width = len(str(count - 1))  # or count, depending on whether you want inclusive
            contact_ids = [f"{i:0{width}d}" for i in range(count)]
            probe_array["contact_ids"] = contact_ids
        if self._contact_array is None:
            self._contact_array = probe_array
        else:
            # Handle the case where the new probe has a different dtype than the existing contact array
            # e.g., one probe has square contacts and the other has circular contacts, so different shape parameters
            existing_dtype = self._contact_array.dtype
            if existing_dtype != probe_dtype:
                fields_to_add = [f for f in probe_dtype.fields if f not in existing_dtype.fields]
                new_dtype = probe_dtype

                # Create a new dtype that is the union of the existing and new dtypes
                new_fields = list(existing_dtype.descr) + [
                    f for f in probe_dtype.descr if f[0] not in existing_dtype.fields
                ]
                new_dtype = np.dtype(new_fields)
                # Create a new array with the new dtype and copy existing data
                new_contact_array = np.zeros(self._contact_array.shape, dtype=new_dtype)
                new_probe_array = np.zeros(probe_array.shape, dtype=new_dtype)
                for name in existing_dtype.names:
                    new_contact_array[name] = self._contact_array[name]
                for name in probe_dtype.names:
                    new_probe_array[name] = probe_array[name]
                self._contact_array = new_contact_array
                probe_array = new_probe_array
            self._contact_array = np.concatenate((self._contact_array, probe_array), axis=0)
        self._probe_contours.append(probe.probe_planar_contour)
        annotations = probe.annotations
        annotations["probe_id"] = probe.annotations.get("probe_id", f"probe_{self._num_probes}")
        self._annotations.append(annotations)
        probe._probe_group = self
        self._num_probes += 1

    def _check_compatible(self, probe: Probe) -> None:
        if probe._probe_group is not None:
            raise ValueError(
                "This probe is already attached to another ProbeGroup. Use probe.copy() to attach it to another ProbeGroup"
            )

        if probe.ndim != self.probes[-1].ndim:
            raise ValueError(
                f"ndim are not compatible: probe.ndim {probe.ndim} " f"!= probegroup ndim {self.probes[-1].ndim}"
            )

        # check global channel maps
        self.check_global_device_wiring_and_ids(new_device_channel_indices=probe.device_channel_indices)

    @property
    def ndim(self) -> int:
        return self.probes[0].ndim

    def copy(self) -> "ProbeGroup":
        """
        Create a copy of the ProbeGroup

        Returns
        -------
        copy: ProbeGroup
            A copy of the ProbeGroup
        """
        copy = ProbeGroup()
        copy._num_probes = self._num_probes
        copy._contact_array = self._contact_array.copy()
        copy._probe_contours = deepcopy(self._probe_contours)
        copy._annotations = deepcopy(self._annotations)
        return copy

    def get_contact_count(self) -> int:
        """
        Total number of channels.

        Returns
        -------
        n: int
            The total number of channels
        """
        n = sum(probe.get_contact_count() for probe in self.probes)
        return n

    def to_numpy(self, complete: bool = False) -> np.ndarray:
        """
        Export all probes into a numpy array.

        Parameters
        ----------
        complete: bool, default: False
            If True, export complete information about the probegroup
            including contact_plane_axes/si_units/device_channel_indices
        """
        if complete:
            return self._contact_array.copy()
        else:
            # Remove fields that are not in the default export
            all_probe_fields = []
            for probe_index in range(self._num_probes):
                probe_fields = self.probes[probe_index].to_numpy(complete=False, probe_index=0).dtype.fields
                for f in probe_fields:
                    if f not in all_probe_fields:
                        all_probe_fields.append(f)
            probe_fields = all_probe_fields

            fields_to_remove = [f for f in self._contact_array.dtype.names if f not in probe_fields]
            dtype = [
                (name, self._contact_array.dtype.fields[name][0])
                for name in self._contact_array.dtype.names
                if name not in fields_to_remove
            ]
            arr = np.zeros(self._contact_array.shape, dtype=dtype)
            for name in arr.dtype.names:
                arr[name] = self._contact_array[name]
            return arr

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "ProbeGroup":
        """Create ProbeGroup from a complex numpy array
        see ProbeGroup.to_numpy()

        Parameters
        ----------
        arr : np.array
            The structured np.array representation of the probe

        Returns
        -------
        probegroup : ProbeGroup
            The instantiated ProbeGroup object
        """
        from .probe import Probe

        probes_indices = np.unique(arr["probe_index"])
        probegroup = ProbeGroup()
        probegroup._contact_array = arr.copy()
        for probe_index in probes_indices:
            probegroup._probe_contours.append(None)
            probegroup._annotations.append({})
        probegroup._num_probes = len(probes_indices)
        return probegroup

    def to_dataframe(self, complete: bool = False) -> "pandas.DataFrame":
        """
        Export the probegroup to a pandas dataframe

        Parameters
        ----------
        complete : bool, default: False
            If True, export complete information about the probegroup,
            including the probe plane axis.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe representation of the probegroup

        """
        import pandas as pd

        df = pd.DataFrame(self.to_numpy(complete=complete))
        df.index = np.arange(df.shape[0], dtype="int64")
        return df

    def to_dict(self, array_as_list: bool = False) -> dict:
        """Create a dictionary of all necessary attributes.

        Parameters
        ----------
        array_as_list : bool, default: False
            If True, arrays are converted to lists, by default False

        Returns
        -------
        d : dict
            The dictionary representation of the probegroup
        """
        d = {}
        d["probes"] = []
        for probe_ind, probe in enumerate(self.probes):
            probe_dict = probe.to_dict(array_as_list=array_as_list)
            d["probes"].append(probe_dict)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ProbeGroup":
        """Instantiate a ProbeGroup from a dictionary

        Parameters
        ----------
        d : dict
            The dictionary representation of the probegroup

        Returns
        -------
        probegroup : ProbeGroup
            The instantiated ProbeGroup object
        """
        probegroup = ProbeGroup()
        for probe_dict in d["probes"]:
            probe = Probe.from_dict(probe_dict)
            probegroup.add_probe(probe)
        return probegroup

    def get_global_device_channel_indices(self) -> np.ndarray:
        """
        Gets the global device channels indices and returns as
        an array

        Returns
        -------
        channels: np.ndarray
            a numpy array vector with 2 columns
            (probe_index, device_channel_indices)

        Notes
        -----
            If a channel within channels has a value of -1 this indicates that that channel
            is disconnected
        """
        total_chan = self.get_contact_count()
        channels = np.zeros(total_chan, dtype=[("probe_index", "int64"), ("device_channel_indices", "int64")])
        channels["probe_index"] = self._contact_array["probe_index"]
        channels["device_channel_indices"] = self._contact_array["device_channel_indices"]
        return channels

    def set_global_device_channel_indices(self, channels: np.ndarray | list) -> None:
        """
        Set global indices for all probes

        Parameters
        ----------
        channels: np.ndarray | list
            The device channal indices to be set
        """
        channels = np.asarray(channels)
        if channels.size != self.get_contact_count():
            raise ValueError(
                f"Wrong channels size {channels.size} for the number of channels {self.get_contact_count()}"
            )
        self._contact_array["device_channel_indices"] = channels

    def get_global_contact_ids(self) -> np.ndarray:
        """
        Gets all contact ids concatenated across probes

        Returns
        -------
        contact_ids: np.ndarray
            An array of the contaact ids across all probes
        """
        contact_ids = self._contact_array["contact_ids"]
        return contact_ids

    def get_global_contact_positions(self) -> np.ndarray:
        """
        Gets all contact positions concatenated across probes

        Returns
        -------
        contact_positions: np.ndarray
            An array of the contact positions across all probes
        """
        contact_positions = np.vstack([probe.contact_positions for probe in self.probes])
        return contact_positions

    def get_slice(self, selection: np.ndarray[bool | int]) -> "ProbeGroup":
        """
        Get a copy of the ProbeGroup with a sub selection of contacts.

        Selection can be boolean or by index

        Parameters
        ----------
        selection : np.array of bool or int (for index)
            Either an np.array of bool or for desired selection of contacts
            or the indices of the desired contacts

        Returns
        -------
        sliced_probe_group: ProbeGroup
            The sliced probe group

        """

        n = self.get_contact_count()

        selection = np.asarray(selection)
        if selection.dtype == "bool":
            assert selection.shape == (
                n,
            ), f"if array of bool given it must be the same size as the number of contacts {selection.shape} != {n}"
            (selection_indices,) = np.nonzero(selection)
        elif selection.dtype.kind == "i":
            assert np.unique(selection).size == selection.size
            if len(selection) > 0:
                assert (
                    0 <= np.min(selection) < n
                ), f"An index within your selection is out of bounds {np.min(selection)}"
                assert (
                    0 <= np.max(selection) < n
                ), f"An index within your selection is out of bounds {np.max(selection)}"
                selection_indices = selection
            else:
                selection_indices = []
        else:
            raise TypeError(f"selection must be bool array or int array, not of type: {type(selection)}")

        if len(selection_indices) == 0:
            return ProbeGroup()

        full_contact_array = self._contact_array
        sliced_contact_array = full_contact_array[selection_indices]
        probe_indices = np.unique(sliced_contact_array["probe_index"])
        new_probe_contours = []
        new_annotations = []
        for new_probe_index, old_probe_index in enumerate(probe_indices):
            sliced_contact_array["probe_index"][
                sliced_contact_array["probe_index"] == old_probe_index
            ] = new_probe_index
            new_probe_contours.append(self._probe_contours[old_probe_index])
            new_annotations.append(self._annotations[old_probe_index])

        sliced_probe_group = ProbeGroup()
        sliced_probe_group._contact_array = sliced_contact_array
        sliced_probe_group._num_probes = len(probe_indices)
        sliced_probe_group._probe_contours = new_probe_contours
        sliced_probe_group._annotations = new_annotations
        return sliced_probe_group

    def check_global_device_wiring_and_ids(self, new_device_channel_indices: np.ndarray | None = None) -> None:
        # check unique device_channel_indices for !=-1
        chans = self.get_global_device_channel_indices()["device_channel_indices"]
        if new_device_channel_indices is not None:
            chans = np.concatenate([chans, new_device_channel_indices])

        keep = chans >= 0
        valid_chans = chans[keep]

        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError("channel device indices are not unique across probes")
