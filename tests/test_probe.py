from probeinterface import Probe
from probeinterface.generator import generate_dummy_probe
from pathlib import Path

import numpy as np

import pytest


def _dummy_position():
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
    positions = _dummy_position()

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_contacts(positions=positions, shapes="square", shape_params={"width": 5})
    probe.set_contacts(positions=positions, shapes="rect", shape_params={"width": 8, "height": 5})

    assert probe.get_contact_count() == 24

    # shape of the probe
    vertices = [(-20, -30), (20, -110), (60, -30), (60, 190), (-20, 190)]
    probe.set_planar_contour(vertices)

    # auto shape (test no error)
    probe.create_auto_shape(probe_type="rect")
    probe.create_auto_shape(probe_type="circular")
    probe.create_auto_shape()

    # annotation
    probe.annotate(manufacturer="me")
    assert "manufacturer" in probe.annotations
    probe.annotate_contacts(impedance=np.random.rand(24) * 1000)
    assert "impedance" in probe.contact_annotations

    # device channel
    chans = np.arange(0, 24, dtype="int")
    np.random.shuffle(chans)
    probe.set_device_channel_indices(chans)

    # contact_ids int or str
    elec_ids = np.arange(24)
    probe.set_contact_ids(elec_ids)
    elec_ids = [f"elec #{e}" for e in range(24)]
    probe.set_contact_ids(elec_ids)

    # copy
    probe2 = probe.copy()

    # move rotate
    probe.move([20, 50])
    probe.rotate(theta=40, center=[0, 0], axis=None)

    # make annimage
    values = np.random.randn(24)
    image, xlims, ylims = probe.to_image(values, method="cubic")

    image2, xlims, ylims = probe.to_image(values, method="cubic", num_pixel=16)

    # ~ from probeinterface.plotting import plot_probe_group, plot_probe
    # ~ import matplotlib.pyplot as plt
    # ~ fig, ax = plt.subplots()
    # ~ plot_probe(probe, ax=ax)
    # ~ ax.imshow(image, extent=xlims+ylims, origin='lower')
    # ~ ax.imshow(image2, extent=xlims+ylims, origin='lower')
    # ~ plt.show()

    # 3d
    probe_3d = probe.to_3d()
    probe_3d.rotate(theta=60, center=[0, 0, 0], axis=[0, 1, 0])

    # 3d-2d
    probe_3d = probe.to_3d()
    probe_2d = probe_3d.to_2d(axes="xz")
    assert np.allclose(probe_2d.contact_positions, probe_3d.contact_positions[:, [0, 2]])

    # ~ from probeinterface.plotting import plot_probe_group, plot_probe
    # ~ import matplotlib.pyplot as plt
    # ~ plot_probe(probe_3d)
    # ~ plt.show()

    # get shanks
    for shank in probe.get_shanks():
        pass
        # print(shank)
        # print(shank.contact_positions)

    # get dict and df
    d = probe.to_dict()
    other = Probe.from_dict(d)

    # export to/from numpy
    arr = probe.to_numpy(complete=False)
    other = Probe.from_numpy(arr)
    arr = probe.to_numpy(complete=True)
    other2 = Probe.from_numpy(arr)
    arr = probe_3d.to_numpy(complete=True)
    other_3d = Probe.from_numpy(arr)

    # export to/from DataFrame
    df = probe.to_dataframe(complete=True)
    other = Probe.from_dataframe(df)
    df = probe.to_dataframe(complete=False)
    other2 = Probe.from_dataframe(df)
    df = probe_3d.to_dataframe(complete=True)
    # print(df.index)
    other_3d = Probe.from_dataframe(df)
    assert other_3d.ndim == 3

    # slice handling
    selection = np.arange(0, 18, 2)
    # print(selection.dtype.kind)
    sliced_probe = probe.get_slice(selection)
    assert sliced_probe.get_contact_count() == 9
    assert sliced_probe.contact_annotations["impedance"].shape == (9,)

    # ~ from probeinterface.plotting import plot_probe_group, plot_probe
    # ~ import matplotlib.pyplot as plt
    # ~ plot_probe(probe)
    # ~ plot_probe(sliced_probe)

    selection = np.ones(24, dtype="bool")
    selection[::2] = False
    sliced_probe = probe.get_slice(selection)
    assert sliced_probe.get_contact_count() == 12
    assert sliced_probe.contact_annotations["impedance"].shape == (12,)

    # ~ plot_probe(probe)
    # ~ plot_probe(sliced_probe)
    # ~ plt.show()


def test_probe_equality_dunder():
    probe1 = generate_dummy_probe()
    probe2 = generate_dummy_probe()

    assert probe1 == probe1
    assert probe2 == probe2
    assert probe1 == probe2

    # Modify probe2
    probe2.move([1, 1])
    assert probe1 != probe2


def test_set_shanks():
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=np.arange(20).reshape(10, 2), shapes="circle", shape_params={"radius": 5})

    # for simplicity each contact is on separate shank
    shank_ids = np.arange(10)
    probe.set_shank_ids(shank_ids)

    assert all(probe.shank_ids == shank_ids.astype(str))


def test_save_to_zarr(tmp_path):
    # Generate a dummy probe instance
    probe = generate_dummy_probe()

    # Define file path in the temporary directory
    folder_path = Path(tmp_path) / "probe.zarr"

    # Save the probe object to Zarr format
    probe.to_zarr(folder_path=folder_path)

    # Reload the probe object from the saved Zarr file
    reloaded_probe = Probe.from_zarr(folder_path=folder_path)

    # Assert that the reloaded probe is equal to the original
    assert probe == reloaded_probe, "Reloaded Probe object does not match the original"


def test_position_uniqueness():
    """Test that the error message matches the full expected string for three duplicates using pytest's match regex."""
    import re

    positions_with_dups = np.array([[0, 0], [10, 10], [0, 0], [20, 20], [0, 0], [10, 10]])
    probe = Probe(ndim=2, si_units="um")
    expected_error = (
        "Contact positions must be unique within a probe. "
        "Found 2 duplicate(s): Position (0, 0) appears at indices [0, 2, 4]; Position (10, 10) appears at indices [1, 5]"
    )

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        probe.set_contacts(positions=positions_with_dups, shapes="circle", shape_params={"radius": 5})


if __name__ == "__main__":
    test_probe()

    tmp_path = Path("tmp")
    tmp_path.mkdir(exist_ok=True)
    test_save_to_zarr(tmp_path)
