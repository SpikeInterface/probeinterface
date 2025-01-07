from pathlib import Path
import numpy as np

import pytest

from probeinterface import read_maxwell
from probeinterface.testing import validate_probe_dict

data_path = Path(__file__).absolute().parent.parent / "data" / "maxwell"


def test_valid_probe_dict():
    file_ = "data.raw.h5"
    probe = read_maxwell(data_path / file_)
    probe_dict = probe.to_dict(array_as_list=True)
    probe_dict["annotations"].update(model_name="placeholder")
    validate_probe_dict(probe_dict)


def test_maxwell():
    """Basic file taken from the ephys data repository and provided by Alessio Buccino"""

    probe = read_maxwell(data_path / "data.raw.h5")

    assert probe.ndim == 2
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 16

    contact_width = 5.45
    contact_height = 9.3
    contact_shape = "rect"
    expected_shape_parameter = {"width": contact_width, "height": contact_height}

    assert np.all(probe.contact_shape_params == expected_shape_parameter)
    assert np.all(probe.contact_shapes == contact_shape)
