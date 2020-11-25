from probeinterface import read_prb

import pytest


def test_prb():
    # TODO fix path for 'fake.prb'
    probebunch = read_prb('fake.prb')
    
    #~ from probeinterface.plotting import plot_probe_bunch
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_bunch(probebunch, with_channel_index=True)
    #~ plt.show()
    
    
    
    
    
if __name__ == '__main__':
    test_prb()