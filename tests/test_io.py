from probeinterface import read_prb, write_prb
from probeinterface import generate_fake_probe, generate_fake_probe_bunch

import pytest


def test_prb():
    # TODO fix path for 'fake.prb'
    probebunch = read_prb('fake.prb')
    
    #~ from probeinterface.plotting import plot_probe_bunch
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_bunch(probebunch, with_channel_index=True)
    #~ plt.show()
    
def test_generate():
    probe = generate_fake_probe()
    probebunch = generate_fake_probe_bunch()



if __name__ == '__main__':
    test_prb()
    test_generate()