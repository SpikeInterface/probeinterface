from probeinterface import ProbeGroup
from probeinterface import generate_dummy_probe

import pytest


def test_probegroup():
    probegroup = ProbeGroup()
    
    for i in range(3):
        probe = generate_dummy_probe()
        probe.move([i*100, i*80])
        probegroup.add_probe(probe)
    
    indices = probegroup.get_global_device_channel_indices()
    ids = probegroup.get_global_electrode_ids()
    
    df = probegroup.to_dataframe()
    #~ print(df['global_electrode_ids'])


def test_probegroup_3d():
    probegroup = ProbeGroup()
    
    for i in range(3):
        probe = generate_dummy_probe().to_3d()
        probe.move([i*100, i*80, i*30])
        probegroup.add_probe(probe)

    assert probegroup.ndim == 3
    
    
    
if __name__ == '__main__':
    test_probegroup()
    test_probegroup_3d()