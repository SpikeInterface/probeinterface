from probeinterface import (generate_dummy_probe, generate_dummy_probe_group,
        generate_tetrode, generate_linear_probe, generate_multi_columns_probe,
        generate_multi_shank)


from pathlib import Path
import numpy as np

import pytest


def test_generate():
    probe = generate_dummy_probe()
    probegroup = generate_dummy_probe_group()
    
    tetrode = generate_tetrode()
    
    multi_columns = generate_multi_columns_probe(num_columns=3,
                num_contact_per_column=[10, 12, 10],
                xpitch=22, ypitch=20,
                y_shift_per_column=[0, -10, 0])
    
    linear = generate_linear_probe(num_elec=16,  ypitch=20,
                    contact_shapes='square', contact_shape_params={'width': 15})
    
    multi_shank = generate_multi_shank()
    
    #~ from probeinterface.plotting import plot_probe_group, plot_probe
    #~ import matplotlib.pyplot as plt
    #~ plot_probe(multi_shank, with_channel_index=True,)
    #~ plt.show()

if __name__ == '__main__':
    test_generate()