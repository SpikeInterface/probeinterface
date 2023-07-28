from pathlib import Path
import numpy as np

import pytest

try:
    import ndx_probeinterface
    import pynwb
except ImportError:
    raise ImportError("Missing `ndx_probeinterface` or `pynwb`")

# data_path = Path(__file__).absolute().parent.parent / "data" / "nwb"

def test_nwb():
    return NotImplemented