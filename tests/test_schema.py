import re

from probeinterface import __version__
from probeinterface.testing import schema


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
