
import pytest

from probeinterface.utils import import_safely

def test_good_import():

    np = import_safely('numpy')
    assert np.__name__ == 'numpy'

def test_handle_import_error():
    with pytest.raises(ImportError):
        import_safely('not_a_real_package')
