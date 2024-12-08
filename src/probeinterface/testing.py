import json
from pathlib import Path

from probeinterface import __version__ as version
import jsonschema

json_schema_file = Path(__file__).absolute().parent.parent.parent / "resources" / "probe.json.schema"
schema = json.load(open(json_schema_file, "r"))


def validate_probe_dict(probe_dict):
    instance = dict(specification="probeinterface", version=version, probes=[probe_dict])
    jsonschema.validate(instance=instance, schema=schema)
