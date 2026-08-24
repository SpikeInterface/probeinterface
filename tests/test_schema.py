import json
import re

import jsonschema

import numpy as np
import pytest

from probeinterface import ProbeGroup, __version__, generate_dummy_probe, write_probeinterface
from probeinterface.testing import schema, validate_probegroup_dict


def test_schema_is_annotated():
    """The schema must carry the annotations documenting what it validates."""
    assert schema.get("$schema"), "The schema must declare the JSON Schema draft it uses ('$schema')."
    assert "specification versions" in schema.get(
        "$comment", ""
    ), "The schema '$comment' must document which specification versions it validates."


def test_package_version_is_compatible_with_schema():
    """
    The current package (specification) version must match the compatibility pattern
    declared by the schema's 'version' property.

    The schema rarely changes, so it stays compatible across many releases. When a
    release introduces an incompatible schema change, bump the 'version' pattern in
    ``src/probeinterface/schema/probe.json.schema`` (e.g. to allow ``1.x.y``).
    """
    pattern = schema["properties"]["version"]["pattern"]
    assert re.fullmatch(pattern, __version__), (
        f"The package version ({__version__}) does not match the schema's declared "
        f"compatibility pattern ({pattern}). Either this is an incompatible schema "
        f"change (update the pattern) or the version is malformed."
    )


def _probegroup(n_probes=3):
    """A ProbeGroup with explicit probe ids."""
    probegroup = ProbeGroup()
    for i in range(n_probes):
        probe = generate_dummy_probe()
        probe.move([i * 100, i * 80])
        probegroup.add_probe(probe, probe_id=f"probe_00{i}")
    return probegroup


def test_probegroup_dict_validates():
    """A ProbeGroup dict, which carries 'probe_ids', validates against the schema."""
    d = _probegroup().to_dict(array_as_list=True)
    assert d["probe_ids"] == ["probe_000", "probe_001", "probe_002"]
    validate_probegroup_dict(d)


def test_probegroup_dict_with_global_contact_order_validates():
    """A reordered ProbeGroup dict, which carries 'global_contact_order', validates."""
    probegroup = _probegroup()
    # an order interleaving contacts across probes, so it is not the natural one
    order = np.concatenate([np.arange(0, 96, 2), np.arange(95, 0, -2)])
    reordered = probegroup.get_slice(order)

    d = reordered.to_dict(array_as_list=True)
    assert d["global_contact_order"] is not None
    validate_probegroup_dict(d)


def test_written_probeinterface_file_validates(tmp_path):
    """The JSON actually written by write_probeinterface validates against the schema."""
    file = tmp_path / "probegroup.json"
    write_probeinterface(file, _probegroup())

    with open(file, "r", encoding="utf8") as f:
        d = json.load(f)
    assert d["specification"] == "probeinterface"
    assert "probe_ids" in d
    validate_probegroup_dict(d)


@pytest.mark.parametrize(
    "probe_ids",
    [["0", "0"], [0, 1], "0"],
    ids=["duplicated", "not_strings", "not_a_list"],
)
def test_invalid_probe_ids_are_rejected(probe_ids):
    d = _probegroup(n_probes=2).to_dict(array_as_list=True)
    d["probe_ids"] = probe_ids
    with pytest.raises(jsonschema.ValidationError):
        validate_probegroup_dict(d)


@pytest.mark.parametrize(
    "global_contact_order",
    [[0, 0], [0.5, 1], [-1, 0], "0"],
    ids=["duplicated", "not_integers", "negative", "not_a_list"],
)
def test_invalid_global_contact_order_is_rejected(global_contact_order):
    d = _probegroup(n_probes=2).to_dict(array_as_list=True)
    d["global_contact_order"] = global_contact_order
    with pytest.raises(jsonschema.ValidationError):
        validate_probegroup_dict(d)


def test_unknown_top_level_key_is_rejected():
    d = _probegroup(n_probes=2).to_dict(array_as_list=True)
    d["unknown_key"] = "unexpected"
    with pytest.raises(jsonschema.ValidationError):
        validate_probegroup_dict(d)
