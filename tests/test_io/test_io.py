from probeinterface import (
    write_probeinterface,
    read_probeinterface,
    write_BIDS_probe,
    read_BIDS_probe,
)
from probeinterface import read_prb, write_prb

from probeinterface import generate_dummy_probe_group, generate_dummy_probe


from pathlib import Path
import numpy as np

import pytest


data_path = Path(__file__).absolute().parent.parent / "data"


def test_probeinterface_format(tmp_path):
    file_path = tmp_path / "test_pi_format.json"
    probegroup = generate_dummy_probe_group()
    write_probeinterface(file_path, probegroup)

    probegroup2 = read_probeinterface(file_path)

    assert len(probegroup.probes) == len(probegroup.probes)

    for i in range(len(probegroup.probes)):
        probe0 = probegroup.probes[i]
        probe1 = probegroup2.probes[i]

        assert probe0.get_contact_count() == probe1.get_contact_count()
        assert np.allclose(probe0.contact_positions, probe1.contact_positions)
        assert np.allclose(probe0.probe_planar_contour, probe1.probe_planar_contour)

        # TODO more test

    # ~ from probeinterface.plotting import plot_probe_group
    # ~ import matplotlib.pyplot as plt
    # ~ plot_probe_group(probegroup, with_contact_id=True, same_axes=False)
    # ~ plot_probe_group(probegroup2, with_contact_id=True, same_axes=False)
    # ~ plt.show()


def test_writeprobeinterface(tmp_path):
    probe = generate_dummy_probe()
    file_path = tmp_path / "test.prb"
    write_probeinterface(file_path, probe)

    probe_read = read_probeinterface(file_path).probes[0]
    assert probe.get_contact_count() == probe_read.get_contact_count()
    assert np.allclose(probe.contact_positions, probe_read.contact_positions)
    assert np.allclose(probe.probe_planar_contour, probe_read.probe_planar_contour)


def test_writeprobeinterface_raises_error_with_bad_input(tmp_path):
    probe = "WrongInput"
    file_path = tmp_path / "test.prb"
    with pytest.raises(TypeError):
        write_probeinterface(file_path, probe)


def test_BIDS_format(tmp_path):
    folder_path = tmp_path / "test_BIDS"
    folder_path.mkdir()

    probegroup = generate_dummy_probe_group()

    # add custom probe type annotation to be
    # compatible with BIDS specifications
    for probe in probegroup.probes:
        probe.annotate(type="laminar")

    # add unique contact ids to be compatible
    # with BIDS specifications
    n_els = sum([p.get_contact_count() for p in probegroup.probes])
    # using np.random.choice to ensure uniqueness of contact ids
    el_ids = np.random.choice(np.arange(1e4, 1e5, dtype="int"), replace=False, size=n_els).astype(str)
    for probe in probegroup.probes:
        probe_el_ids, el_ids = np.split(el_ids, [probe.get_contact_count()])
        probe.set_contact_ids(probe_el_ids)

        # switch to more generic dtype for shank_ids
        probe.set_shank_ids(probe.shank_ids.astype(str))

    write_BIDS_probe(folder_path, probegroup)

    probegroup_read = read_BIDS_probe(folder_path)

    # compare written (original) and read probegroup
    assert len(probegroup.probes) == len(probegroup_read.probes)
    for probe_orig, probe_read in zip(probegroup.probes, probegroup_read.probes):
        # check that all attributes are preserved
        # check all old annotations are still present
        assert probe_orig.annotations.items() <= probe_read.annotations.items()
        # check if the same attribute lists are present (independent of order)
        assert len(probe_orig.contact_ids) == len(probe_read.contact_ids)
        assert all(np.isin(probe_orig.contact_ids, probe_read.contact_ids))

        # the transformation of contact order between the two probes
        t = np.array([list(probe_read.contact_ids).index(elid) for elid in probe_orig.contact_ids])

        assert all(probe_orig.contact_ids == probe_read.contact_ids[t])
        assert all(probe_orig.shank_ids == probe_read.shank_ids[t])
        assert all(probe_orig.contact_shapes == probe_read.contact_shapes[t])
        assert probe_orig.ndim == probe_read.ndim
        assert probe_orig.si_units == probe_read.si_units

        for i in range(len(probe_orig.probe_planar_contour)):
            assert all(probe_orig.probe_planar_contour[i] == probe_read.probe_planar_contour[i])
        for sid, shape_params in enumerate(probe_orig.contact_shape_params):
            assert shape_params == probe_read.contact_shape_params[t][sid]
        for i in range(len(probe_orig.contact_positions)):
            assert all(probe_orig.contact_positions[i] == probe_read.contact_positions[t][i])
        for i in range(len(probe.contact_plane_axes)):
            for dim in range(len(probe.contact_plane_axes[i])):
                assert all(probe_orig.contact_plane_axes[i][dim] == probe_read.contact_plane_axes[t][i][dim])


def test_BIDS_format_empty(tmp_path):
    folder_path = tmp_path / "test_BIDS_minimal"
    folder_path.mkdir()
    # create empty BIDS probe and contact files
    probes_path = folder_path / "probes.tsv"
    with open(probes_path, "w") as f:
        f.write("probe_id\ttype")

    contacts_path = folder_path / "contacts.tsv"
    with open(contacts_path, "w") as f:
        f.write("contact_id\tprobe_id")

    read_BIDS_probe(folder_path)


def test_BIDS_format_minimal(tmp_path):
    folder_path = tmp_path / "test_BIDS_minimal"
    folder_path.mkdir()
    # create minimal BIDS probe and contact files
    probes_path = folder_path / "probes.tsv"
    with open(probes_path, "w") as f:
        f.write("probe_id\ttype\n" "0\tcustom\n" "1\tgeneric")

    contacts_path = folder_path / "contacts.tsv"
    with open(contacts_path, "w") as f:
        f.write("contact_id\tprobe_id\n" "01\t0\n" "02\t0\n" "11\t1\n" "12\t1")

    probegroup = read_BIDS_probe(folder_path)

    assert len(probegroup.probes) == 2

    for pid, probe in enumerate(probegroup.probes):
        assert probe.get_contact_count() == 2
        assert probe.annotations["probe_id"] == str(pid)
        assert probe.annotations["type"] == ["custom", "generic"][pid]
        assert all(probe.contact_ids == [["01", "02"], ["11", "12"]][pid])


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


def test_prb(tmp_path):
    probegroup = read_prb(data_path / "dummy.prb")

    with open("two_tetrodes.prb", "w") as f:
        f.write(prb_two_tetrodes)

    two_tetrode = read_prb("two_tetrodes.prb")
    assert len(two_tetrode.probes) == 2
    assert two_tetrode.probes[0].get_contact_count() == 4
    file_path = tmp_path / "two_tetrodes_written.prb"
    write_prb(file_path, two_tetrode)
    two_tetrode_back = read_prb(file_path)

    # ~ from probeinterface.plotting import plot_probe_group
    # ~ import matplotlib.pyplot as plt
    # ~ plot_probe_group(probegroup, with_contact_id=True, same_axes=False)
    # ~ plt.show()

    # from probeinterface.plotting import plot_probe
    # import matplotlib.pyplot as plt
    # plot_probe(probe)
    # plt.show()


if __name__ == "__main__":
    # test_probeinterface_format()
    # test_BIDS_format()
    # test_BIDS_format_empty()
    # test_BIDS_format_minimal()
    pass
