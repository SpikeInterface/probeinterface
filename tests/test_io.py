from probeinterface import write_probeinterface, read_probeinterface
from probeinterface import read_prb, write_prb
from probeinterface import read_spikeglx
from probeinterface import generate_dummy_probe_bunch


from pathlib import Path
import numpy as np

import pytest


folder = Path(__file__).absolute().parent



def test_probeinterface_format():
    filename = 'test_pi_format.json'
    probegroup = generate_dummy_probe_bunch()
    write_probeinterface(filename, probegroup)
    
    probegroup2 = read_probeinterface(filename)
    
    assert len(probegroup.probes) == len(probegroup.probes)
    
    for i in range(len(probegroup.probes)):
        probe0 = probegroup.probes[i]
        probe1 = probegroup2.probes[i]
        
        assert probe0.get_electrode_count() == probe1.get_electrode_count()
        assert np.allclose(probe0.electrode_positions,probe1.electrode_positions)
        assert np.allclose(probe0.probe_shape_vertices,probe1.probe_shape_vertices)
        
        # TODO more test

    #~ from probeinterface.plotting import plot_probe_bunch
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_bunch(probegroup, with_channel_index=True, same_axe=False)
    #~ plot_probe_bunch(probegroup2, with_channel_index=True, same_axe=False)
    #~ plt.show()

    

prb_two_tetrodes = """
channel_groups = {
    0: {
            'channels' : [0,1,2,3],
            'geometry': {
                0: [0, 50],
                1: [50, 0],
                2: [0, -50],
                3: [-50, 0],
            }
    },
    1: {
            'channels' : [4,5,6,7],
            'geometry': {
                4: [0, 50],
                5: [50, 0],
                6: [0, -50],
                7: [-50, 0],
            }
    }
}
"""

def test_prb():
    probegroup = read_prb(folder / 'dummy.prb')
    
    with open('two_tetrodes.prb', 'w') as f:
        f.write(prb_two_tetrodes)
    
    two_tetrode = read_prb('two_tetrodes.prb')
    assert len(two_tetrode.probes) == 2
    assert two_tetrode.probes[0].get_electrode_count() == 4
    
    write_prb('two_tetrodes_written.prb', two_tetrode)
    two_tetrode_back = read_prb('two_tetrodes_written.prb')
    
    
    
    
    
    #~ from probeinterface.plotting import plot_probe_bunch
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_bunch(probegroup, with_channel_index=True, same_axe=False)
    #~ plt.show()


def test_readspikeglx():
    probe = read_spikeglx(folder / 'Noise_g0_t0.imec0.ap.meta')

    #~ from probeinterface.plotting import plot_probe_bunch, plot_probe
    #~ import matplotlib.pyplot as plt
    #~ plot_probe(probe)
    #~ plt.show()
    
    
    


if __name__ == '__main__':
    test_probeinterface_format()
    test_prb()
    test_readspikeglx()
