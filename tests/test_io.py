from probeinterface import write_probeinterface, read_probeinterface, write_BIDS_probe, read_BIDS_probe
from probeinterface import read_prb, write_prb
from probeinterface import read_spikeglx, read_openephys, read_imro, write_imro
from probeinterface import generate_dummy_probe_group


from pathlib import Path
import numpy as np

import pytest


folder = Path(__file__).absolute().parent



def test_probeinterface_format():
    filename = 'test_pi_format.json'
    probegroup = generate_dummy_probe_group()
    write_probeinterface(filename, probegroup)
    
    probegroup2 = read_probeinterface(filename)
    
    assert len(probegroup.probes) == len(probegroup.probes)
    
    for i in range(len(probegroup.probes)):
        probe0 = probegroup.probes[i]
        probe1 = probegroup2.probes[i]
        
        assert probe0.get_contact_count() == probe1.get_contact_count()
        assert np.allclose(probe0.contact_positions,probe1.contact_positions)
        assert np.allclose(probe0.probe_planar_contour,probe1.probe_planar_contour)
        
        # TODO more test

    #~ from probeinterface.plotting import plot_probe_group
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_group(probegroup, with_channel_index=True, same_axes=False)
    #~ plot_probe_group(probegroup2, with_channel_index=True, same_axes=False)
    #~ plt.show()


def test_BIDS_format():
    folder = Path('test_BIDS')
    folder.mkdir(exist_ok=True)
    probegroup = generate_dummy_probe_group()

    # add custom probe type annotation to be
    # compatible with BIDS specifications
    for probe in probegroup.probes:
        probe.annotate(type='laminar')

    # add unique contact ids to be compatible
    # with BIDS specifications
    n_els = sum([p.get_contact_count() for p in probegroup.probes])
    # using np.random.choice to ensure uniqueness of contact ids
    el_ids = np.random.choice(np.arange(1e4, 1e5, dtype='int'),
                              replace=False, size=n_els).astype(str)
    for probe in probegroup.probes:
        probe_el_ids, el_ids = np.split(el_ids, [probe.get_contact_count()])
        probe.set_contact_ids(probe_el_ids)

        # switch to more generic dtype for shank_ids
        probe.set_shank_ids(probe.shank_ids.astype(str))

    write_BIDS_probe(folder, probegroup)

    probegroup_read = read_BIDS_probe(folder)

    # compare written (original) and read probegroup
    assert len(probegroup.probes) == len(probegroup_read.probes)
    for probe_orig, probe_read in zip(probegroup.probes,
                                      probegroup_read.probes):
        # check that all attributes are preserved
        # check all old annotations are still present
        assert probe_orig.annotations.items() <= probe_read.annotations.items()
        # check if the same attribute lists are present (independent of order)
        assert len(probe_orig.contact_ids) == len(probe_read.contact_ids)
        assert all(np.in1d(probe_orig.contact_ids, probe_read.contact_ids))

        # the transformation of contact order between the two probes
        t = np.array([list(probe_read.contact_ids).index(elid)
                              for elid in probe_orig.contact_ids])

        assert all(probe_orig.contact_ids == probe_read.contact_ids[t])
        assert all(probe_orig.shank_ids == probe_read.shank_ids[t])
        assert all(probe_orig.contact_shapes == probe_read.contact_shapes[t])
        assert probe_orig.ndim == probe_read.ndim
        assert probe_orig.si_units == probe_read.si_units

        for i in range(len(probe_orig.probe_planar_contour)):
            assert all(probe_orig.probe_planar_contour[i] ==
                       probe_read.probe_planar_contour[i])
        for sid, shape_params in enumerate(probe_orig.contact_shape_params):
            assert shape_params == probe_read.contact_shape_params[t][sid]
        for i in range(len(probe_orig.contact_positions)):
            assert all(probe_orig.contact_positions[i] ==
                       probe_read.contact_positions[t][i])
        for i in range(len(probe.contact_plane_axes)):
            for dim in range(len(probe.contact_plane_axes[i])):
                assert all(probe_orig.contact_plane_axes[i][dim] ==
                           probe_read.contact_plane_axes[t][i][dim])


def test_BIDS_format_empty():
    folder = Path('test_BIDS_minimal')
    folder.mkdir(exist_ok=True)

    # create empty BIDS probe and contact files
    with open(folder.joinpath('probes.tsv'), 'w') as f:
        f.write('probe_id\ttype')

    with open(folder.joinpath('contacts.tsv'), 'w') as f:
        f.write('contact_id\tprobe_id')

    read_BIDS_probe(folder)


def test_BIDS_format_minimal():
    folder = Path('test_BIDS_minimal')
    folder.mkdir(exist_ok=True)

    # create minimal BIDS probe and contact files
    with open(folder.joinpath('probes.tsv'), 'w') as f:
        f.write('probe_id\ttype\n'
                '0\tcustom\n'
                '1\tgeneric')

    with open(folder.joinpath('contacts.tsv'), 'w') as f:
        f.write('contact_id\tprobe_id\n'
                '01\t0\n'
                '02\t0\n'
                '11\t1\n'
                '12\t1')

    probegroup = read_BIDS_probe(folder)

    assert len(probegroup.probes) == 2

    for pid, probe in enumerate(probegroup.probes):
        assert probe.get_contact_count() == 2
        assert probe.annotations['probe_id'] == str(pid)
        assert probe.annotations['type'] == ['custom', 'generic'][pid]
        assert all(probe.contact_ids == [['01', '02'], ['11', '12']][pid])

    

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
    assert two_tetrode.probes[0].get_contact_count() == 4
    
    write_prb('two_tetrodes_written.prb', two_tetrode)
    two_tetrode_back = read_prb('two_tetrodes_written.prb')
    
    
    
    
    
    #~ from probeinterface.plotting import plot_probe_group
    #~ import matplotlib.pyplot as plt
    #~ plot_probe_group(probegroup, with_channel_index=True, same_axes=False)
    #~ plt.show()


def test_readspikeglx():
    # NP1
    probe = read_spikeglx(folder / 'Noise_g0_t0.imec0.ap.meta')
    print(probe)
    print(probe.contact_ids)

    # NP2 4 shanks
    probe = read_spikeglx(folder / 'TEST_20210920_0_g0_t0.imec0.ap.meta')
    print(probe)
    print(probe.contact_ids)

    # NP2 1 shanks
    probe = read_spikeglx(folder / 'p2_g0_t0.imec0.ap.meta')
    print(probe)
    print(probe.contact_ids)


def test_readopenephys():
    # NP1
    probe = read_openephys(folder / "OE_Neuropix-PXI" / "settings.xml")

    # multiple probes
    probeA = read_openephys(folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
                            probe_name="ProbeA")
    print(probeA)
    print(probeA.contact_ids)
    probeB = read_openephys(folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
                            stream_name="RecordNode#ProbeB")
    print(probeB)
    print(probeB.contact_ids)
    probeC = read_openephys(folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
                            serial_number="20403311714")
    print(probeC)
    print(probeC.contact_ids)
    probeD = read_openephys(folder / "OE_Neuropix-PXI-multi-probe" / "settings.xml",
                            probe_name="ProbeD")
    print(probeD)
    print(probeD.contact_ids)
    assert probeA.annotations['probe_serial_number'] == "17131307831"
    assert probeB.annotations['probe_serial_number'] == "20403311724"
    assert probeC.annotations['probe_serial_number'] == "20403311714"
    assert probeD.annotations['probe_serial_number'] == "21144108671"

    # from probeinterface.plotting import plot_probe_group, plot_probe
    # import matplotlib.pyplot as plt
    # plot_probe(probe, with_contact_id=True)
    # plt.show()

def test_readimro():
    probe = read_imro(folder / "test_multi_shank.imro")
    write_imro(folder/"multi_shank_written.imro", probe)
    probe2 = read_imro(folder / "multi_shank_written.imro")
    np.testing.assert_array_equal(probe2.contact_ids , probe.contact_ids)
    np.testing.assert_array_equal(probe2.contact_positions, probe.contact_positions)

if __name__ == '__main__':
    # test_probeinterface_format()
    # test_BIDS_format()
    # test_BIDS_format_empty()
    # test_BIDS_format_minimal()
    
    # test_prb()
    test_readspikeglx()
    test_readopenephys()
    
