import numpy as np


class ProbeGroup:
    """
    Class to handlesa group of Probe objects and the wiring to a device.
    
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
        Total number of channel.
        """
        n = sum(probe.get_contact_count() for probe in self.probes)
        return n

    def get_global_device_channel_indices(self):
        """
        return a numpy array vector with 2 columns
        (probe_index, device_channel_index)
        
        Note:
            channel -1 means not connected
        """
        total_chan = self.get_channel_count()
        channels = np.zeros(total_chan, dtype=[('probe_index', 'int64'), ('device_channel_index', 'int64')])
        channels['device_channel_index'] = -1

        ind = 0
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            channels['probe_index'][ind:ind + n] = i
            if probe.device_channel_indices is not None:
                channels['device_channel_index'][ind:ind + n] = probe.device_channel_indices
            ind += n

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

        all_ids = []
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            ids = probe.contact_ids
            if ids is None:
                ids = [''] * n
            all_ids.append(ids)
        all_ids = np.concatenate(all_ids)
        return all_ids

    def check_global_device_wiring_and_ids(self):
        # check unique device_channel_indices for !=-1
        chans = self.get_global_device_channel_indices()
        keep = chans['device_channel_index'] >= 0
        valid_chans = chans[keep]['device_channel_index']

        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError('channel device index are not unique across probes')

        # check unique ids for != ''
        all_ids = self.get_global_contact_ids()
        keep = [e != '' for e in all_ids]
        valid_ids = all_ids[keep]

        if valid_ids.size != np.unique(valid_ids).size:
            raise ValueError('contact_ids are not unique across probes')
    
    
    def to_dataframe(self):
        import pandas as pd
        
        all_df =[]
        for i, probe in enumerate(self.probes):
            df = probe.to_dataframe()
            df['probe_num'] = i
            df.index = [(i, ind) for ind in df.index]
            all_df.append(df)
        df = pd.concat(all_df, axis=0)
        
        df['global_contact_ids'] = self.get_global_contact_ids()
        
        return df
    
    def get_groups(self, group_mode='by_probe'):
        """
        Get sub groups of channels  by contacts or by shank.
        This used for spike sorting in spikeinterface.
        
        Parameters
        ----------
        group_mode: 'by_probe' or ''by_shank'
        
        Returns
        -----
        
        
        """
        assert group_mode in ('by_probe', 'by_shank')

        positions = []
        device_indices = []
        if group_mode == 'by_probe':
            for probe in self.probes:
                positions.append(probe.contact_positions)
                device_indices.append(probe.device_channel_indices)
        elif group_mode == 'by_shank':
            for probe in self.probes:
                for shank in probe.get_shanks():
                    positions.append(shank.contact_positions)
                    device_indices.append(shank.device_channel_indices)
        
        return positions, device_indices
