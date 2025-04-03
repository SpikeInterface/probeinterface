import json
from pathlib import Path

from probeinterface import __version__ as version

json_schema_file = Path(__file__).absolute().parent / "schema" / "probe.json.schema"
schema = json.load(open(json_schema_file, "r"))


def validate_probe_dict(probe_dict):
    import jsonschema

    instance = dict(specification="probeinterface", version=version, probes=[probe_dict])
    jsonschema.validate(instance=instance, schema=schema)
