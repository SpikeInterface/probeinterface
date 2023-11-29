from __future__ import annotations
import numpy as np
from .utils import generate_unique_ids
from .probe import Probe


class ProbeGroup:
    """
    Class to handle a group of Probe objects and the global wiring to a device.

    Optionally, it can handle the location of different probes.

    """

    def __init__(self):
        self.probes = []

    def add_probe(self, probe: Probe):
        """
        Add an additional probe to the ProbeGroup

        Parameters
        ----------
        probe: Probe
            The probe to add to the ProbeGroup

        """
        if len(self.probes) > 0:
            self._check_compatible(probe)

        self.probes.append(probe)
        probe._probe_group = self

    def _check_compatible(self, probe: Probe):
        if probe._probe_group is not None:
            raise ValueError(
                "This probe is already attached to another ProbeGroup. Use probe.copy() to attach it to another ProbeGroup"
            )

        if probe.ndim != self.probes[-1].ndim:
            raise ValueError(
                f"ndim are not compatible: probe.ndim {probe.ndim} " f"!= probegroup ndim {self.probes[-1].ndim}"
            )

        # check global channel maps
        self.probes.append(probe)
        self.check_global_device_wiring_and_ids()
        self.probes = self.probes[:-1]

    @property
    def ndim(self):
        return self.probes[0].ndim

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

        fields = []
        probe_arr = []

        # loop over probes to get all fields
        dtype = [("probe_index", "int64")]
        fields = []
        for probe_index, probe in enumerate(self.probes):
            arr = probe.to_numpy(complete=complete)
            probe_arr.append(arr)
            for k in arr.dtype.fields:
                if k not in fields:
                    fields.append(k)
                    dtype += [(k, arr.dtype.fields[k][0])]

        pg_arr = []
        for probe_index, probe in enumerate(self.probes):
            arr = probe_arr[probe_index]
            arr_ext = np.zeros(probe.get_contact_count(), dtype=dtype)
            arr_ext["probe_index"] = probe_index
            for k in fields:
                if k in arr.dtype.fields:
                    arr_ext[k] = arr[k]
            pg_arr.append(arr_ext)

        pg_arr = np.concatenate(pg_arr, axis=0)
        return pg_arr

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
        for probe_index in probes_indices:
            mask = arr["probe_index"] == probe_index
            probe = Probe.from_numpy(arr[mask])
            probegroup.add_probe(probe)
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

    def to_dict(self, array_as_list: bool = False):
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
    def from_dict(d: dict):
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
        arr = self.to_numpy(complete=True)
        channels["probe_index"] = arr["probe_index"]
        channels["device_channel_indices"] = arr["device_channel_indices"]
        return channels

    def set_global_device_channel_indices(self, channels: np.array | list):
        """
        Set global indices for all probes

        Parameters
        ----------
        channels: np.array | list
            The device channal indices to be set
        """
        channels = np.asarray(channels)
        if channels.size != self.get_contact_count():
            raise ValueError(
                f"Wrong channels size {channels.size} for the number of channels {self.get_contact_count()}"
            )

        # first reset previsous indices
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices([-1] * n)

        # then set new indices
        ind = 0
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices(channels[ind : ind + n])
            ind += n

    def get_global_contact_ids(self) -> np.ndarray:
        """
        Gets all contact ids concatenated across probes

        Returns
        -------
        contact_ids: np.ndarray
            An array of the contaact ids across all probes
        """
        contact_ids = self.to_numpy(complete=True)["contact_ids"]
        return contact_ids

    def check_global_device_wiring_and_ids(self):
        # check unique device_channel_indices for !=-1
        chans = self.get_global_device_channel_indices()
        keep = chans["device_channel_indices"] >= 0
        valid_chans = chans[keep]["device_channel_indices"]

        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError("channel device indices are not unique across probes")

    def auto_generate_probe_ids(self, *args, **kwargs):
        """
        Annotate all probes with unique probe_id values.

        Parameters
        ----------
        *args: will be forwarded to `probeinterface.utils.generate_unique_ids`
        **kwargs: will be forwarded to
            `probeinterface.utils.generate_unique_ids`
        """

        if any("probe_id" in p.annotations for p in self.probes):
            raise ValueError("Probe already has a `probe_id` annotation.")

        if not args:
            args = 1e7, 1e8
        # 3rd argument has to be the number of probes
        args = args[:2] + (len(self.probes),)

        # creating unique probe ids in case probes do not have any yet
        probe_ids = generate_unique_ids(*args, **kwargs).astype(str)
        for pid, probe in enumerate(self.probes):
            probe.annotate(probe_id=probe_ids[pid])

    def auto_generate_contact_ids(self, *args, **kwargs):
        """
        Annotate all contacts with unique contact_id values.

        Parameters
        ----------
        *args: will be forwarded to `probeinterface.utils.generate_unique_ids`
        **kwargs: will be forwarded to
            `probeinterface.utils.generate_unique_ids`
        """

        if not args:
            args = 1e7, 1e8
        # 3rd argument has to be the number of probes
        args = args[:2] + (self.get_contact_count(),)

        contact_ids = generate_unique_ids(*args, **kwargs).astype(str)

        for probe in self.probes:
            el_ids, contact_ids = np.split(contact_ids, [probe.get_contact_count()])
            probe.set_contact_ids(el_ids)
