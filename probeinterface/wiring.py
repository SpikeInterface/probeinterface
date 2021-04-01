"""
Here a module set automatically the `Probe.device_channel_indices` field.
"""
import numpy as np

pathways = {
    # this is the neuronexus H32 with omnetics connected to the intantec RHD headstage
    'H32>RHD2132': [
        16, 17, 18, 20, 21, 22, 31, 30, 29, 27, 26, 25, 24, 28, 23, 19,
        12, 8, 3, 7, 6, 5, 4, 2, 1, 0, 9, 10, 11, 13, 14, 15],

    'ASSY-156>RHD2164': [
        46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33,
        30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
        62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 49]
}


def get_available_pathways():
    return list(pathways.keys())


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
