import json
from pathlib import Path

from probeinterface import __version__ as version

json_schema_file = Path(__file__).absolute().parent / "schema" / "probe.json.schema"
with open(json_schema_file, "r") as f:
    schema = json.load(f)


def validate_probe_dict(probe_dict):
    import jsonschema

    instance = dict(specification="probeinterface", version=version, probes=[probe_dict])
    jsonschema.validate(instance=instance, schema=schema)


def validate_probegroup_dict(probegroup_dict):
    """
    Validate a full ProbeGroup dict (as returned by ``ProbeGroup.to_dict()``) against
    the schema. The "specification" and "version" keys are added if missing, so that
    both a raw ``to_dict()`` and the content of a probeinterface JSON file can be passed.
    """
    import jsonschema

    instance = {"specification": "probeinterface", "version": version, **probegroup_dict}
    jsonschema.validate(instance=instance, schema=schema)
