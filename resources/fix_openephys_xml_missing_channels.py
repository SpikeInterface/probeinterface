"""
Script to detect and fix missing channels in OpenEphys XML settings files.

This script corrects XML files where channels are missing from the
CHANNELS, ELECTRODE_XPOS, and ELECTRODE_YPOS tags. It fills in missing
channels by inferring their values based on existing patterns in the file.

Typical usage example:
    python fix_openephys_xml_missing_channels.py --file_path settings.xml --overwrite --verbose
"""
import argparse
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from probeinterface.utils import import_safely


def fix_openephys_xml_file(
    file_path: Union[str, Path],
    overwrite: bool = True,
    verbose: bool = False
):
    """
    Fix missing channels in an OpenEphys XML settings file.

    This function parses the XML file, detects missing channels in the
    CHANNELS, ELECTRODE_XPOS, and ELECTRODE_YPOS tags, and fills them in
    by inferring values from existing data patterns.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the XML file to be fixed.
    overwrite : bool, optional
        If True, overwrite the original file. If False, save as .fixed.xml.
        Default is True.
    verbose : bool, optional
        If True, print detailed information about the process.
        Default is False.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If unable to infer fill values for missing channels.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse the XML file
    ET = import_safely("xml.etree.ElementTree")
    tree = ET.parse(str(file_path))
    root = tree.getroot()

    # Find all relevant elements
    channels_elements = root.findall(".//CHANNELS")
    xpos_elements = root.findall(".//ELECTRODE_XPOS")
    ypos_elements = root.findall(".//ELECTRODE_YPOS")

    for channels, xpos, ypos in zip(channels_elements, xpos_elements, ypos_elements):
        channel_names = np.array(list(channels.attrib.keys()))
        channel_ids = np.array([int(ch[2:]) for ch in channel_names])
        sorted_channel_ids = sorted(channel_ids)
        all_channel_ids = set(range(sorted_channel_ids[0], sorted_channel_ids[-1] + 1))
        missing_channels = sorted(all_channel_ids - set(sorted_channel_ids))

        if not missing_channels:
            if verbose:
                print("No missing channels detected.")
            continue

        warnings.warn(f"Missing channels detected in XML: {missing_channels}")

        # Detect repeating pattern for <ELECTRODE_XPOS>
        xpos_values = [int(value) for value in xpos.attrib.values()]
        pattern_length = next(
            (i for i in range(1, len(xpos_values) // 2) if xpos_values[:i] == xpos_values[i:2 * i]),
            len(xpos_values)
        )
        xpos_pattern = xpos_values[:pattern_length]

        # Detect step for <ELECTRODE_YPOS>
        ypos_values = [int(value) for value in ypos.attrib.values()]
        unique_steps = np.unique(np.diff(sorted(set(ypos_values))))
        if len(unique_steps) != 1:
            raise ValueError("Unable to determine unique step size for ELECTRODE_YPOS.")
        ypos_step = unique_steps[0]

        # Fill in missing channels
        for channel_id in missing_channels:
            # Find the closest channel before or after
            prev_channels = [ch for ch in channel_ids if ch < channel_id]
            next_channels = [ch for ch in channel_ids if ch > channel_id]

            if prev_channels:
                nearest_channel_id = max(prev_channels)
            elif next_channels:
                nearest_channel_id = min(next_channels)
            else:
                raise ValueError(f"Cannot find reference channel for missing channel {channel_id}")

            channel_fill_value = channels.attrib[f"CH{nearest_channel_id}"]
            channels.set(f"CH{channel_id}", channel_fill_value)

            xpos_fill_value = xpos_pattern[channel_id % pattern_length]
            xpos.set(f"CH{channel_id}", str(xpos_fill_value))

            ypos_fill_value = (channel_id // 2) * ypos_step
            ypos.set(f"CH{channel_id}", str(ypos_fill_value))

    if not overwrite:
        file_path = file_path.with_suffix(".fixed.xml")

    # Save the updated XML
    tree.write(file_path)
    if verbose:
        print(f"Fixed XML file saved to: {file_path}")


def main():
    """
    Command-line interface for fixing OpenEphys XML files.

    Parses command-line arguments and calls fix_openephys_xml_file.
    """
    parser = argparse.ArgumentParser(description="Fix missing channels in OpenEphys XML settings files.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the XML file to fix.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original file.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information.")
    args = parser.parse_args()

    fix_openephys_xml_file(
        file_path=args.file_path,
        overwrite=args.overwrite,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
