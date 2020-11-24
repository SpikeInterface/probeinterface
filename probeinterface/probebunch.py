


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
    
    