


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
    
    def _check_compatible(self, probe):
        if probe.ndim != self.probes[-1].ndim:
            raise ValueError("ndim are not compatible")
    
    @property
    def ndim(self):
        return self.probes[0].ndim
    
    def get_channel_count(self):
        """
        Total number of channel.
        """
        n = sum(probe.get_electrode_count() for probe in self.probes)
        return n
    
    