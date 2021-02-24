"""
Here a module set automatically the `Probe.device_channel_indices` field.
"""
import numpy as np

pathways = {
    # this is the neuronexus H32 with omnetics connected to the intantec RHD headstage
    'H32>RHD': [
        16, 17, 18, 20, 21, 22, 31, 30, 29, 27, 26, 25, 24, 28, 23, 19,
        12, 8, 3, 7, 6, 5, 4, 2, 1, 0, 9, 10, 11, 13, 14, 15],
}


def wire_probe(probe, pathway, channel_offset=0):
    """
    Inplace wiring for a Probe.
    """
    assert pathway in pathways
    chan_indices = np.array(pathways[pathway], dtype='int64') + channel_offset
    assert chan_indices.size == probe.get_contact_count()
    probe.set_device_channel_indices(chan_indices)


if __name__ == '__main__':

    for pathway, chan_indices in pathways.items():
        chan_indices = np.array(chan_indices)
        print(pathway, chan_indices.size)
        assert np.unique(chan_indices).size == chan_indices.size
