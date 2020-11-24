


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
        self.probes.append(probe)
    
    def get_channel_count(self):
        """
        Total number of channel.
        """
        n = sum(probe.get_electrode_count() for probe in self.probes)
        return n
    
    