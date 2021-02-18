"""
This provide function to download and cache locally some probe files
from some manufacturers.

The library is hosted here: 
https://gin.g-node.org/spikeinterface/probeinterface_library

The 

gin platform enable  contribution from users.

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
    Download in the cache directory the probeinterface file based on JSON.
    Note that the file is itself a ProbeGroup but on the repo each file
    represent one probe.
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
