from pathlib import Path
import numpy as np

import pytest

from probeinterface import read_3brain

data_path = Path(__file__).absolute().parent.parent / "data" / "3brain"


def test_3brain():
    """Files on ephy_test_data"""

    contact_shape = "square"

    for file, contact_width, contact_pitch, contact_count in [
        # old brw3.x format with default 42 um pitch (even if wrong)
        ("biocam_hw3.0_fw1.6.brw", 21, 42, 36),
        # new brw4.x format has chip information included
        ("biocam_hw3.0_fw1.7.0.12_raw.brw", 21, 60, 100),
    ]:
        probe = read_3brain(data_path / file)

        assert probe.ndim == 2
        assert probe.get_shank_count() == 1
        assert probe.get_contact_count() == contact_count
        assert np.all(probe.contact_shapes == contact_shape)
        assert np.all(probe.contact_shape_params == {"width": contact_width})
        unique_rows = np.unique(probe.contact_positions[:, 0])
        assert np.all(np.isclose(np.diff(unique_rows), contact_pitch)), file
        unique_cols = np.unique(probe.contact_positions[:, 0])
        assert np.all(np.isclose(np.diff(unique_cols), contact_pitch))
