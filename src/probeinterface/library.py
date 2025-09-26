"""
Provides functions to download and cache pre-existing probe files
from some manufacturers.

The library is hosted here:
https://gin.g-node.org/spikeinterface/probeinterface_library

The gin platform enables contributions from users.

"""

from __future__ import annotations
import os
from pathlib import Path
from urllib.request import urlopen
import requests
from typing import Optional

from .io import read_probeinterface

# OLD URL on gin
# public_url = "https://web.gin.g-node.org/spikeinterface/probeinterface_library/raw/master/"

# Now on github since 2023/06/15
public_url = "https://raw.githubusercontent.com/SpikeInterface/probeinterface_library/main/"

# check this for windows and osx
cache_folder = Path(os.path.expanduser("~")) / ".config" / "probeinterface" / "library"


def download_probeinterface_file(manufacturer: str, probe_name: str):
    """Download the probeinterface file to the cache directory.
    Note that the file is itself a ProbeGroup but on the repo each file
    represents one probe.

    Parameters
    ----------
    manufacturer : "cambridgeneurotech" | "neuronexus" | "plexon" | "imec" | "sinaps"
        The probe manufacturer
    probe_name : str (see probeinterface_libary for options)
        The probe name
    """
    os.makedirs(cache_folder / manufacturer, exist_ok=True)
    localfile = cache_folder / manufacturer / (probe_name + ".json")
    distantfile = public_url + f"{manufacturer}/{probe_name}/{probe_name}.json"
    dist = urlopen(distantfile)
    with open(localfile, "wb") as f:
        f.write(dist.read())


def get_from_cache(manufacturer: str, probe_name: str) -> Optional["Probe"]:
    """
    Get Probe from local cache

    Parameters
    ----------
    manufacturer : "cambridgeneurotech" | "neuronexus" | "plexon" | "imec" | "sinaps"
        The probe manufacturer
    probe_name : str (see probeinterface_libary for options)
        The probe name

    Returns
    -------
    probe : Probe object, or None if no probeinterface JSON file is found

    """

    localfile = cache_folder / manufacturer / (probe_name + ".json")
    if not localfile.is_file():
        return None
    else:
        probegroup = read_probeinterface(localfile)
        probe = probegroup.probes[0]
        probe._probe_group = None
        return probe


def get_probe(manufacturer: str, probe_name: str, name: Optional[str] = None) -> "Probe":
    """
    Get probe from ProbeInterface library

    Parameters
    ----------
    manufacturer : "cambridgeneurotech" | "neuronexus" | "plexon" | "imec" | "sinaps"
        The probe manufacturer
    probe_name : str (see probeinterface_libary for options)
        The probe name
    name : str | None, default: None
        Optional name for the probe

    Returns
    ----------
    probe : Probe object

    """

    probe = get_from_cache(manufacturer, probe_name)

    if probe is None:
        download_probeinterface_file(manufacturer, probe_name)
        probe = get_from_cache(manufacturer, probe_name)
    if probe.manufacturer == "":
        probe.manufacturer = manufacturer
    if name is not None:
        probe.name = name

    return probe


def get_manufacturers_in_library(tag=None) -> list[str]:
    """
    Get the list of available manufacturers in the library

    Returns
    -------
    manufacturers : list of str
        List of available manufacturers
    """
    return list_github_folders("SpikeInterface", "probeinterface_library", ref=tag)


def get_probes_in_library(manufacturer: str, tag=None) -> list[str]:
    """
    Get the list of available probes for a given manufacturer

    Parameters
    ----------
    manufacturer : str
        The probe manufacturer

    Returns
    -------
    probes : list of str
        List of available probes for the given manufacturer
    """
    return list_github_folders("SpikeInterface", "probeinterface_library", path=manufacturer, ref=tag)


def get_tags_in_library() -> list[str]:
    """
    Get the list of available tags in the library

    Returns
    -------
    tags : list of str
        List of available tags
    """
    tags = []
    tags = get_all_tags("SpikeInterface", "probeinterface_library")
    return tags


### UTILS
def get_latest_tag(owner: str, repo: str, token: str = None):
    """
    Get the latest tag (by order returned from GitHub) for a repo.
    Returns the tag name, or None if no tags exist.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/tags"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API returned {resp.status_code}: {resp.text}")
    tags = resp.json()
    if not tags:
        return None
    return tags[0]["name"]  # first entry is the latest


def get_all_tags(owner: str, repo: str, token: str = None):
    """
    Get all tags for a repo.
    Returns a list of tag names, or an empty list if no tags exist.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/tags"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API returned {resp.status_code}: {resp.text}")
    tags = resp.json()
    return [tag["name"] for tag in tags]


def list_github_folders(owner: str, repo: str, path: str = "", ref: str = None, token: str = None):
    """
    Return a list of directory names in the given repo at the specified path.
    You can pass a branch, tag, or commit SHA via `ref`.
    If token is provided, use it for authenticated requests (higher rate limits).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {}
    if ref:
        params["ref"] = ref
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API returned status {resp.status_code}: {resp.text}")
    items = resp.json()
    return [item["name"] for item in items if item.get("type") == "dir" and item["name"][0] != "."]
