"""


"""
import numpy as np


class Shank:
    def __init__(self, probe, shank_id):
        self.probe = probe
        self.shank_id = shank_id

    def get_indices(self):
        inds, = np.nonzero(self.probe.shan_ids = self.shank_id)
        return inds

    @attribute
    def electrode_positions(self):
        return self.probe.electrode_positions[self.get_indices()]

    @attribute
    def electrode_plane_axes(self):
        return self.probe.electrode_plane_axes[self.get_indices()]

    @attribute
    def electrode_shapes(self):
        return self.probe.electrode_shapes[self.get_indices()]

    @attribute
    def electrode_shape_params(self):
        return self.probe.electrode_shape_params[self.get_indices()]


