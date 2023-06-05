"""
Automatically set the `Probe.device_channel_indices` field.
"""
import numpy as np

pathways = {
    # this is the neuronexus H32 with omnetics connected to the intantec RHD headstage
    'H32>RHD2132': [
        16, 17, 18, 20, 21, 22, 31, 30, 29, 27, 26, 25, 24, 28, 23, 19,
        12, 8, 3, 7, 6, 5, 4, 2, 1, 0, 9, 10, 11, 13, 14, 15],

    'ASSY-156>RHD2164': [
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48],

    # Mapping the ASSY-116>RHD2132
    'ASSY-116>RHD2132': [
        24, 23, 25, 22, 26, 21, 27, 20, 28, 19, 29, 18, 30, 17, 31, 16,  0,
        15,  1, 14,  2, 13,  3, 12,  4, 11,  5, 10,  6,  9,  7,  8],

    # nicely given by Aaron Wrong
    'ASSY-77>Adpt.A64-Om32_2x-sm-NN>RHD2164': [
        62, 63, 60, 61, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
        32, 33, 34, 35, 37, 36, 39, 41, 43, 45, 47, 38, 42, 44, 46, 40,
        22, 16, 18, 20, 24, 17, 19, 21, 23, 25, 26, 27, 29, 28, 31, 30,
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 3, 2, 1, 0],

    # done by Samuel Garcia, Alessio Buccino, Jessie Goins for Pierre-Pascal Lenck-Santini
    'ASSY-77>Adpt.A64-Om32_2x-sm-NN>two_RHD2132': [
        47, 32, 46, 33, 34, 45, 35, 44, 36, 43, 37, 42, 38, 41, 39, 40,
        16, 31, 17, 30, 29, 18, 28, 27, 26, 25, 24, 19, 21, 22, 23, 20,
        11, 8, 9, 10, 12, 7, 6, 5, 4, 3, 13, 2, 1, 14, 0, 15,
        55, 56, 54, 57, 53, 58, 52, 59, 51, 60, 50, 61, 62, 49, 63, 48],

    # from PDF documention of mini-amp-64 page 5
    'cambridgeneurotech_mini-amp-64': [
        # connector J2 TOP
        41, 39, 38, 37, 35, 34, 33, 32, 29, 30, 28, 26, 25, 24, 22, 20,
        46, 45, 44, 43, 42, 40, 36, 31, 27, 23, 21, 18, 19, 17, 16, 14,
        # connector J1 BOTTOM
        55, 53, 54, 52, 51, 50, 49, 48, 47, 15, 13, 12, 11, 9, 10, 8,
        63, 62, 61, 60, 59, 58, 57, 56, 7, 6, 5, 4, 3, 2, 1, 0
    ]
}


def get_available_pathways():
    """Return available pathways

    Returns
    -------
    pathway : list
        The list of available pathways
    """
    return list(pathways.keys())


def wire_probe(probe, pathway, channel_offset=0):
    """Inplace wiring for a Probe using a pathway

    Parameters
    ----------
    probe : Probe
        The probe to wire
    pathway : str
        The pathway to use
    channel_offset : int, optional
        An optional offset to add to the device_channel_indices, by default 0
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
