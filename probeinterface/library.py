"""
Provides functions to download and cache pre-existing probe files
from some manufacturers.

The library is hosted here:
https://gin.g-node.org/spikeinterface/probeinterface_library

The gin platform enables contributions from users.

"""
import os
from pathlib import Path
from urllib.request import urlopen

from .io import read_probeinterface

public_url = "https://web.gin.g-node.org/spikeinterface/probeinterface_library/raw/master/"

# check this for windows and osx
cache_folder = Path(os.path.expanduser("~")) / '.config' / 'probeinterface' / 'library'


def download_probeinterface_file(manufacturer, probe_name):
    """
    Download the probeinterface file to the cache directory.
    Note that the file is itself a ProbeGroup but on the repo each file
    represents one probe.

    """

    os.makedirs(cache_folder / manufacturer, exist_ok=True)
    localfile = cache_folder / manufacturer / (probe_name + '.json')
    distantfile = public_url + f'{manufacturer}/{probe_name}/{probe_name}.json'
    dist = urlopen(distantfile)
    with open(localfile, 'wb') as f:
        f.write(dist.read())


def get_from_cache(manufacturer, probe_name):
    """
    Get Probe from cache.

    Returns
    -------
    probe : Probe object, or None if no probeinterface JSON file is found

    """

    localfile = cache_folder / manufacturer / (probe_name + '.json')
    if not localfile.is_file():
        return None
    else:
        probegroup = read_probeinterface(localfile)
        probe = probegroup.probes[0]
        probe._probe_group = None
        return probe


def get_probe(manufacturer, probe_name):
    """
    Get probe

    Parameters
    ----------
    manufacturer: str

    probe_name: str

    Returns
    ----------
    Probe object

    """

    probe = get_from_cache(manufacturer, probe_name)

    if probe is None:
        download_probeinterface_file(manufacturer, probe_name)
        probe = get_from_cache(manufacturer, probe_name)

    return probe
