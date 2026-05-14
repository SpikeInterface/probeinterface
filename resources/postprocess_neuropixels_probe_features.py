"""
Post-process neuropixels_probe_features.json after syncing from ProbeTable.

Derives two mappings from the catalogue and writes them back into the JSON:

- z_imro_format_type_to_imro_format: IMRO type code -> IMRO format name
  (e.g. "0" -> "imro_np1000", "1110" -> "imro_np1110")

- z_imro_format_type_to_part_number: IMRO type code -> canonical probe part number
  (e.g. "0" -> "NP1000", "1110" -> "NP1110")

This script is called by the GitHub Action workflow that syncs probe_features.json
from billkarsh/ProbeTable, and can also be run standalone.
"""

import json
import re
from pathlib import Path

PROBE_FEATURES_PATH = (
    Path(__file__).absolute().parent
    / "../src/probeinterface/resources/neuropixels_probe_features.json"
)


def _parse_type_values_from_val_def(val_def: str) -> list[str]:
    """Extract IMRO type code(s) from a val_def string.

    Two patterns in ProbeTable:
      type:{0,1020,1030,...}  -> set of values
      type:1110               -> single value
    """
    match = re.match(r"type:\{([^}]+)\}", val_def)
    if match:
        return [v.strip() for v in match.group(1).split(",")]

    match = re.match(r"type:(\d+)", val_def)
    if match:
        return [match.group(1)]

    raise ValueError(f"Cannot parse type from val_def: {val_def!r}")


def build_derived_mappings(probe_features: dict) -> tuple[dict, dict]:
    """Build type-to-format and type-to-part-number mappings from the catalogue."""

    imro_formats = probe_features["z_imro_formats"]
    probes = probe_features["neuropixels_probes"]

    # 1. Build type -> format mapping from val_def entries
    type_to_format = {}
    for key, val_def in imro_formats.items():
        if not key.endswith("_val_def"):
            continue
        # e.g. "imro_np1000_val_def" -> "imro_np1000"
        format_name = key.removesuffix("_val_def")
        for type_code in _parse_type_values_from_val_def(val_def):
            if type_code in type_to_format:
                raise ValueError(
                    f"IMRO type {type_code!r} maps to both "
                    f"{type_to_format[type_code]!r} and {format_name!r}"
                )
            type_to_format[type_code] = format_name

    # 2. Build type -> canonical part number mapping
    #    For each type, find probes that use the matching format, then pick
    #    the first NP-prefixed part number alphabetically.
    #
    #    We also need to verify the candidate actually belongs to this type,
    #    not just the same format. For example, NP1021 uses imro_np1000 format
    #    but its IMRO type is not "0". We filter by checking the format's
    #    val_def includes the type code we're resolving.

    # Invert: format -> set of type codes it covers
    format_to_types = {}
    for type_code, format_name in type_to_format.items():
        format_to_types.setdefault(format_name, set()).add(type_code)

    type_to_part_number = {}
    for type_code, format_name in sorted(type_to_format.items()):
        candidates = [
            pn
            for pn, spec in probes.items()
            if spec.get("imro_table_format_type") == format_name
        ]

        # Prefer a probe whose part number contains the type code (e.g. NP1020 for type "1020").
        # This matters because many probes share the same IMRO format but have different
        # physical geometries (e.g. NP1000 has 960 contacts, NP1020 has 2496).
        exact_matches = sorted(
            pn for pn in candidates if pn.startswith("NP") and type_code in pn
        )
        if exact_matches:
            type_to_part_number[type_code] = exact_matches[0]
            continue

        # Fall back to first NP-prefixed name alphabetically
        np_candidates = sorted(pn for pn in candidates if pn.startswith("NP"))
        other_candidates = sorted(pn for pn in candidates if not pn.startswith("NP"))
        ordered = np_candidates + other_candidates

        if ordered:
            type_to_part_number[type_code] = ordered[0]

    return type_to_format, type_to_part_number


def postprocess(filepath: Path = PROBE_FEATURES_PATH) -> None:
    filepath = filepath.resolve()
    with open(filepath) as f:
        probe_features = json.load(f)

    type_to_format, type_to_part_number = build_derived_mappings(probe_features)

    probe_features["z_imro_format_type_to_imro_format"] = dict(sorted(type_to_format.items(), key=lambda kv: int(kv[0])))
    probe_features["z_imro_format_type_to_part_number"] = dict(sorted(type_to_part_number.items(), key=lambda kv: int(kv[0])))

    with open(filepath, "w") as f:
        json.dump(probe_features, f, indent=4)
        f.write("\n")

    print(f"Wrote derived mappings to {filepath}")
    print(f"  z_imro_format_type_to_imro_format: {len(type_to_format)} entries")
    print(f"  z_imro_format_type_to_part_number: {len(type_to_part_number)} entries")
    for type_code in sorted(type_to_format, key=int):
        pn = type_to_part_number.get(type_code, "???")
        print(f"    type {type_code:>5s} -> format={type_to_format[type_code]}, part_number={pn}")


if __name__ == "__main__":
    postprocess()
