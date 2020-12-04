from probeinterface import (generate_dummy_probe, generate_dummy_probe_bunch,
        generate_tetrode, generate_linear_probe, generate_multi_columns_probe)


from pathlib import Path
import numpy as np

import pytest


def test_generate():
    probe = generate_dummy_probe()
    probebunch = generate_dummy_probe_bunch()
    
    tetrode = generate_tetrode()
    
    multi_columns = generate_multi_columns_probe(num_columns=3,
                num_elec_per_column=[10, 12, 10],
                xpitch=22, ypitch=20,
                y_shift_per_column=[0, -10, 0])
    
    linear = generate_linear_probe(num_elec=16,  ypitch=20,
                    electrode_shapes='square', electrode_shape_params={'width': 15})
    
    #~ from probeinterface.plotting import plot_probe_bunch, plot_probe
    #~ import matplotlib.pyplot as plt
    #~ plot_probe(linear, with_channel_index=True,)
    #~ plt.show()

if __name__ == '__main__':
    test_generate()