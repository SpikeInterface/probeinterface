from probeinterface import ProbeGroup
from probeinterface import generate_dummy_probe

import pytest

import numpy as np


def test_probegroup():
    probegroup = ProbeGroup()
    
    nchan = 0
    for i in range(3):
        probe = generate_dummy_probe()
        probe.move([i*100, i*80])
        n = probe.get_contact_count()
        probe.set_device_channel_indices(np.arange(n)[::-1] + nchan)
        shank_ids = np.ones(n)
        shank_ids[:n//2] *= i * 2
        shank_ids[n//2:] *= i * 2 +1 
        probe.set_shank_ids(shank_ids)
        probegroup.add_probe(probe)
        
        
        nchan += n
    
    indices = probegroup.get_global_device_channel_indices()
    
    ids = probegroup.get_global_contact_ids()
    
    df = probegroup.to_dataframe()
    #~ print(df['global_contact_ids'])
    
    arr = probegroup.to_numpy(complete=False)
    other = ProbeGroup.from_numpy(arr)
    arr = probegroup.to_numpy(complete=True)
    other = ProbeGroup.from_numpy(arr)
    
    d = probegroup.to_dict()
    other = ProbeGroup.from_dict(d)

    #~ from probeinterface.plotting import plot_probe_group, plot_probe
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_group(probegroup)
    #~ plot_probe_group(other)
    #~ plt.show()
    
    # checking automatic generation of ids with new dummy probes
    probegroup.probes = []
    for i in range(3):
        probegroup.add_probe(generate_dummy_probe())
    probegroup.auto_generate_contact_ids()
    probegroup.auto_generate_probe_ids()

    for p in probegroup.probes:
        assert p.contact_ids is not None
        assert 'probe_id' in p.annotations

def test_probegroup_3d():
    probegroup = ProbeGroup()
    
    for i in range(3):
        probe = generate_dummy_probe().to_3d()
        probe.move([i*100, i*80, i*30])
        probegroup.add_probe(probe)

    assert probegroup.ndim == 3
    

if __name__ == '__main__':
    test_probegroup()
    #~ test_probegroup_3d()