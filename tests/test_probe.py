from probeinterface import Probe

import numpy as np

import pytest

def _dummy_posistion():
    n = 24
    positions = np.zeros((n, 2))
    for i in range(n):
        x = i // 8
        y = i % 8
        positions[i] = x, y
    positions *= 20
    positions[8:16, 1] -= 10
    return positions
    

def test_probe():
    positions = _dummy_posistion()
    
    probe = Probe(ndim=2, si_units='um')
    probe.set_electrodes(positions=positions, shapes='circle', shape_params={'radius': 5})
    probe.set_electrodes(positions=positions, shapes='square', shape_params={'width': 5})
    probe.set_electrodes(positions=positions, shapes='rect', shape_params={'width': 8, 'height':5 })

    assert probe.get_electrode_count() == 24

    # shape of the probe
    vertices = [(-20, -30), (20, -110), (60, -30), (60, 190), (-20, 190)]
    probe.set_planar_contour(vertices)
        
    # auto shape
    probe.create_auto_shape()
    
    # device channel
    chans = np.arange(0, 24, dtype='int')
    np.random.shuffle(chans)
    probe.set_device_channel_indices(chans)
    
    # electrode_ids int or str
    elec_ids = np.arange(24)
    probe.set_electrode_ids(elec_ids)
    elec_ids = [f'elec #{e}' for e in range(24)]
    probe.set_electrode_ids(elec_ids)
    
    # copy
    probe2 = probe.copy()
    
    # move rotate
    probe.move([20, 50])
    probe.rotate(theta=40, center=[0, 0], axis=None)

    # make annimage
    values = np.random.randn(24)
    image, xlims, ylims = probe.to_image(values, method='cubic')
    
    image2, xlims, ylims = probe.to_image(values, method='cubic', num_pixel=16)
    
    #~ from probeinterface.plotting import plot_probe_group, plot_probe
    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ plot_probe(probe, ax=ax)
    #~ ax.imshow(image, extent=xlims+ylims, origin='lower')
    #~ ax.imshow(image2, extent=xlims+ylims, origin='lower')
    #~ plt.show()
    
    
    # 3d
    probe_3d = probe.to_3d()
    probe_3d.rotate(theta=60, center=[0, 0, 0], axis=[0,1,0])

    #~ from probeinterface.plotting import plot_probe_group, plot_probe
    #~ import matplotlib.pyplot as plt
    #~ plot_probe(probe_3d)
    #~ plt.show()

    # get shanks
    for shank in probe.get_shanks():
        print(shank)
        print(shank.electrode_positions)
        
    # get dict and df
    d = probe.to_dict()
    #~ print(d)
    df = probe.to_dataframe()
    print(df)



if __name__ == '__main__':
    test_probe()
