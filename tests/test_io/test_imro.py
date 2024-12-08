import glob
from pathlib import Path

import pytest
import numpy as np

from probeinterface import read_imro, write_imro
from probeinterface.testing import validate_probe_dict

data_path = Path(__file__).absolute().parent.parent / "data" / "imro"
imro_files = glob.glob(str(data_path / "*.imro"))

imro_files.pop(imro_files.index(str(data_path / "test_non_standard.imro")))


@pytest.mark.parametrize("file_", imro_files)
def test_valid_probe_dict(file_: str):
    probe = read_imro(data_path / file_)
    validate_probe_dict(probe.to_dict(array_as_list=True))


def test_reading_multishank_imro(tmp_path):
    probe = read_imro(data_path / "test_multi_shank.imro")

    file_path = tmp_path / "multi_shank_written.imro"
    write_imro(file_path, probe)
    probe2 = read_imro(file_path)
    np.testing.assert_array_equal(probe2.contact_ids, probe.contact_ids)
    np.testing.assert_array_equal(probe2.contact_positions, probe.contact_positions)


def test_reading_imro_2_single_shank(tmp_path):
    probe = read_imro(data_path / "test_single_shak_2.0.imro")

    file_path = tmp_path / "test_single_shak_2.0_written.imro"
    write_imro(file_path, probe)
    probe2 = read_imro(file_path)
    np.testing.assert_array_equal(probe2.contact_ids, probe.contact_ids)
    np.testing.assert_array_equal(probe2.contact_positions, probe.contact_positions)


def test_reading_old_imro(tmp_path):
    probe = read_imro(data_path / "test_old_probe.imro")

    file_path = tmp_path / "test_old_probe.imro"
    write_imro(file_path, probe)
    probe2 = read_imro(file_path)
    np.testing.assert_array_equal(probe2.contact_ids, probe.contact_ids)
    np.testing.assert_array_equal(probe2.contact_positions, probe.contact_positions)


def test_raising_error_when_writing_with_wrong_type(tmp_path):
    probe = read_imro(data_path / "test_old_probe.imro")

    file_path = tmp_path / "test_old_probe.imro"
    probe.annotations["probe_type"] = "something_that_should_make_write_fail"
    with pytest.raises(ValueError):
        write_imro(file_path, probe)


def test_non_standard_file():
    with pytest.raises(ValueError):
        probe = read_imro(data_path / "test_non_standard.imro")


if __name__ == "__main__":
    test_reading_old_imro(Path("tmp"))
