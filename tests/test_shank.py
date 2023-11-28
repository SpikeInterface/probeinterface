import pytest
import numpy as np

from probeinterface import generate_multi_shank


@pytest.fixture(scope="module")
def testing_shank():
    num_shank = 1
    num_columns = 1
    num_contact_per_column = 6
    contact_shapes = "square"
    contact_shape_params = {'width': 6}
    multi_shank_probe = generate_multi_shank(
        num_shank=num_shank,
        num_columns=num_columns,
        num_contact_per_column=num_contact_per_column,
        contact_shapes=contact_shapes,
        contact_shape_params=contact_shape_params
    )
    shank = multi_shank_probe.get_shanks()[0]

    return shank


def test_channel_indices(testing_shank):
    expected_channel_indices = np.arange(6)
    assert np.allclose(testing_shank.get_indices(), expected_channel_indices)


def test_contact_count(testing_shank):
    expected_contact_count = 6
    assert testing_shank.get_contact_count() == expected_contact_count


def test_contact_positions(testing_shank):
    probe = testing_shank.probe
    expected_contact_positions = probe.contact_positions[testing_shank.get_indices()]
    assert np.allclose(testing_shank.contact_positions, expected_contact_positions)


def test_contact_plane_axes(testing_shank):
    # Shank should have the same contact plane axes as the probe
    plane_axes_probe = testing_shank.probe.contact_plane_axes
    plane_axes_shank = testing_shank.contact_plane_axes

    assert np.allclose(plane_axes_probe, plane_axes_shank)


def test_contact_shapes(testing_shank):
    probe = testing_shank.probe
    expected_contact_shapes = probe.contact_shapes[testing_shank.get_indices()]
    assert np.array_equal(testing_shank.contact_shapes, expected_contact_shapes)


def test_contact_shape_parameters(testing_shank):
    probe = testing_shank.probe
    expected_contact_shape_params = probe.contact_shape_params[
        testing_shank.get_indices()
    ]
    assert np.array_equal(
        testing_shank.contact_shape_params, expected_contact_shape_params
    )


def test_device_channel_indices(testing_shank):
    probe = testing_shank.probe
    assert testing_shank.device_channel_indices is None
