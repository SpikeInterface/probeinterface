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

    def add_probe(self, probe):
        """

        """
        if len(self.probes) > 0:
            self._check_compatible(probe)

        self.probes.append(probe)
        probe._probe_group = self

    def _check_compatible(self, probe):
        if probe._probe_group is not None:
            raise ValueError("This probe is already attached to another ProbeGroup")

        if probe.ndim != self.probes[-1].ndim:
            raise ValueError("ndim are not compatible")

        # check global channel maps
        self.probes.append(probe)
        self.check_global_device_wiring_and_ids()
        self.probes = self.probes[:-1]

    @property
    def ndim(self):
        return self.probes[0].ndim

    def get_channel_count(self):
        """
        Total number of channels.
        """
        n = sum(probe.get_contact_count() for probe in self.probes)
        return n

    def to_numpy(self, complete=False):
        """
        Export all probes into a numpy array.
        """

        fields = []
        probe_arr = []

        # loop over probes to get all fields
        dtype = [('probe_index', 'int64')]
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
            arr_ext['probe_index'] = probe_index
            for k in fields:
                if k in arr.dtype.fields:
                    arr_ext[k] = arr[k]
            pg_arr.append(arr_ext)

        pg_arr = np.concatenate(pg_arr, axis=0)
        return pg_arr

    @staticmethod
    def from_numpy(arr):
        from .probe import Probe
        probes_indices = np.unique(arr['probe_index'])
        probegroup = ProbeGroup()
        for probe_index in probes_indices:
            mask = arr['probe_index'] == probe_index
            probe = Probe.from_numpy(arr[mask])
            probegroup.add_probe(probe)
        return probegroup

    def to_dataframe(self, complete=False):
        import pandas as pd
        df = pd.DataFrame(self.to_numpy(complete=complete))
        df.index = np.arange(df.shape[0], dtype='int64')
        return df

    def to_dict(self, array_as_list=False):
        """Create a dictionary of all necessary attributes.

        Parameters
        ----------
        array_as_list : bool, optional
            If True, arrays are converted to lists, by default False

        Returns
        -------
        d : dict
            The dictionary representation of the probegroup
        """
        d = {}
        d['probes'] = []
        for probe_ind, probe in enumerate(self.probes):
            probe_dict = probe.to_dict(array_as_list=array_as_list)
            d['probes'].append(probe_dict)
        return d

    @staticmethod
    def from_dict(d):
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
        for probe_dict in d['probes']:
            probe = Probe.from_dict(probe_dict)
            probegroup.add_probe(probe)
        return probegroup

    def get_global_device_channel_indices(self):
        """
        return a numpy array vector with 2 columns
        (probe_index, device_channel_indices)

        Note:
            channel -1 means not connected
        """
        total_chan = self.get_channel_count()
        channels = np.zeros(total_chan, dtype=[('probe_index', 'int64'), ('device_channel_indices', 'int64')])
        arr = self.to_numpy(complete=True)
        channels['probe_index'] = arr['probe_index']
        channels['device_channel_indices'] = arr['device_channel_indices']
        return channels

    def set_global_device_channel_indices(self, channels):
        """
        Set global indices for all probes
        """
        channels = np.asarray(channels)
        if channels.size != self.get_channel_count():
            raise ValueError('Wrong channels size')

        # first reset previsous indices
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices([-1] * n)

        # then set new indices
        ind = 0
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices(channels[ind:ind + n])
            ind += n

    def get_global_contact_ids(self):
        """
        get all contact ids concatenated across probes
        """
        contact_ids = self.to_numpy(complete=True)['contact_ids']
        return contact_ids

    def check_global_device_wiring_and_ids(self):
        # check unique device_channel_indices for !=-1
        chans = self.get_global_device_channel_indices()
        keep = chans['device_channel_indices'] >= 0
        valid_chans = chans[keep]['device_channel_indices']

        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError('channel device index are not unique across probes')

        # check unique ids for != ''
        all_ids = self.get_global_contact_ids()
        keep = [e != '' for e in all_ids]
        valid_ids = all_ids[keep]

        if valid_ids.size != np.unique(valid_ids).size:
            raise ValueError('contact_ids are not unique across probes')

    def auto_generate_probe_ids(self, *args, **kwargs):
        """
        Annotate all probes with unique probe_id values.

        Parameters
        ----------
        *args: will be forwarded to `probeinterface.utils.generate_unique_ids`
        **kwargs: will be forwarded to
            `probeinterface.utils.generate_unique_ids`
        """

        if any('probe_id' in p.annotations for p in self.probes):
            raise ValueError('Probe does already have a `probe_id` annotation.')

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

        if any(p.contact_ids is not None for p in self.probes):
            raise ValueError('Some contacts already have contact ids '
                             'assigned.')

        if not args:
            args = 1e7, 1e8
        # 3rd argument has to be the number of probes
        args = args[:2] + (self.get_channel_count(),)

        contact_ids = generate_unique_ids(*args, **kwargs).astype(str)

        for probe in self.probes:
            el_ids, contact_ids = np.split(contact_ids,
                                             [probe.get_contact_count()])
            probe.set_contact_ids(el_ids)

