import numpy as np


class Shank:
    """
    Class to handle one shank within a probe.

    A Shank object is a sub-component of a Probe object.

    """
    def __init__(self, probe, shank_id):
        self.probe = probe
        self.shank_id = shank_id

    def get_indices(self):
        inds, = np.nonzero(self.probe.shank_ids == self.shank_id)
        return inds

    def get_contact_count(self):
        return self.get_indices().size

    @property
    def contact_positions(self):
        return self.probe.contact_positions[self.get_indices()]

    @property
    def contact_plane_axes(self):
        return self.probe.contact_plane_axes[self.get_indices()]

    @property
    def contact_shapes(self):
        return self.probe.contact_shapes[self.get_indices()]

    @property
    def contact_shape_params(self):
        return self.probe.contact_shape_params[self.get_indices()]

    @property
    def device_channel_indices(self):
        return self.probe.device_channel_indices[self.get_indices()]

