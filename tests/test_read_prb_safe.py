"""Class guard for the probeinterface read_prb() exec RCE (CWE-95)."""

import os, re, inspect
from probeinterface import read_prb
import probeinterface.io as pio


def test_read_prb_does_not_execute(tmp_path):
    marker = tmp_path / "pwned.txt"
    p = tmp_path / "x.prb"
    p.write_text(
        "import os\n"
        f"os.system('echo x> {marker.as_posix()}')\n"
        "channel_groups = {0: {'channels': [0], 'geometry': {0: [0.0, 0.0]}}}\n"
    )
    pg = read_prb(str(p))
    assert not marker.exists(), "read_prb executed code from the .prb file (CWE-95)"
    assert pg is not None and len(pg.probes) == 1  # channel_groups still parsed


def test_read_prb_source_has_no_exec_eval():
    src = inspect.getsource(pio.read_prb)
    assert not re.search(r"(?<![\w.])exec\(", src), "read_prb still uses exec()"
    assert not re.search(r"(?<![\w.])eval\(", src), "read_prb still uses eval()"
