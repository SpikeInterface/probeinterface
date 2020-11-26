import numpy as np


class ProbeBunch:
    """
    Class that handle a bunch of Probe and the wiring to device.
    
    Optionaly handle the geometry in between probes.
    
    """
    def __init__(self):
        self.probes = []
    
    def add_probe(self, probe):
        """
        
        """
        if len(self.probes) > 0:
            self._check_compatible(probe)
            
        self.probes.append(probe)
        probe._probe_bunch = self
    
    def _check_compatible(self, probe):
        if probe._probe_bunch is not None:
            raise ValueError("This probe is already attached to another ProbeBunch")
        
        if probe.ndim != self.probes[-1].ndim:
            raise ValueError("ndim are not compatible")
        
        # check global channel maps
        self.probes.append(probe)
        self.check_global_device_wiring()
        self.probes =self.probes[:-1]
        
    
    @property
    def ndim(self):
        return self.probes[0].ndim
    
    def get_channel_count(self):
        """
        Total number of channel.
        """
        n = sum(probe.get_electrode_count() for probe in self.probes)
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
            n = probe.get_electrode_count()
            channels['probe_index'][ind:ind+n] = i
            if probe.device_channel_indices is not None:
                channels['device_channel_index'][ind:ind+n] = probe.device_channel_indices
            ind += n
        
        return channels
    
    def check_global_device_wiring(self):
        chans = self.get_global_device_channel_indices()
        keep = chans['device_channel_index'] >=0
        valid_chans = chans[keep]['device_channel_index']
        
        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError('channel are not unique on device across probes')

