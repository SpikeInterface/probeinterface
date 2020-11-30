from probeinterface import ProbeBunch
from probeinterface import generate_fake_probe

import pytest


def test_probebunch():
    probebunch = ProbeBunch()
    
    for i in range(3):
        probe = generate_fake_probe()
        probe.move([i*100, i*80])
        probebunch.add_probe(probe)
    
    indices = probebunch.get_global_device_channel_indices()
    ids = probebunch.get_global_electrode_ids()


def test_probebunch_3d():
    probebunch = ProbeBunch()
    
    for i in range(3):
        probe = generate_fake_probe().to_3d()
        probe.move([i*100, i*80, i*30])
        probebunch.add_probe(probe)

    assert probebunch.ndim == 3
    
    
    
if __name__ == '__main__':
    test_probebunch()
    test_probebunch_3d()