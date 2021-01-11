import numpy as np


class Shank:
    """
    Class to handle a shank in probe.

    A Shank object is a sub-part of a Probe object.

    """
    def __init__(self, probe, shank_id):
        self.probe = probe
        self.shank_id = shank_id

    def get_indices(self):
        inds, = np.nonzero(self.probe.shank_ids == self.shank_id)
        return inds
        
    def get_electrode_count(self):
        return self.get_indices().size

    @property
    def electrode_positions(self):
        return self.probe.electrode_positions[self.get_indices()]

    @property
    def electrode_plane_axes(self):
        return self.probe.electrode_plane_axes[self.get_indices()]

    @property
    def electrode_shapes(self):
        return self.probe.electrode_shapes[self.get_indices()]

    @property
    def electrode_shape_params(self):
        return self.probe.electrode_shape_params[self.get_indices()]


