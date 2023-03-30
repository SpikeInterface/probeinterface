from pathlib import Path
import numpy as np

import pytest

from probeinterface import read_3brain

data_path = Path(__file__).absolute().parent.parent / "data" / "3brain"


def test_3brain():
    """Basic file taken from the ephys data and provided by Robert Wolff and Valter Tucci"""

    probe = read_3brain(data_path / "biocam_hw3.0_fw1.6.brw")

    assert probe.ndim == 2
    assert probe.get_shank_count() == 1
    assert probe.get_contact_count() == 36

    contact_width = 21
    contact_shape = "square"

    assert np.all(probe.contact_shapes == contact_shape)
    assert np.all(probe.contact_shape_params == {"width": contact_width})
