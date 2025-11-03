"""
Provides functions to download and cache pre-existing probe files
from some manufacturers.

The library is hosted here:
https://gin.g-node.org/spikeinterface/probeinterface_library

The gin platform enables contributions from users.

"""

from __future__ import annotations
import os
import warnings
from pathlib import Path
from urllib.request import urlopen
import requests
from typing import Optional

from .io import read_probeinterface

# OLD URL on gin
# public_url = "https://web.gin.g-node.org/spikeinterface/probeinterface_library/raw/master/"
# Now on github since 2023/06/15
public_url = "https://raw.githubusercontent.com/SpikeInterface/probeinterface_library/"


# check this for windows and osx
def get_cache_folder() -> Path:
    """Get the cache folder for probeinterface library files.

    Returns
    -------
    cache_folder : Path
        The path to the cache folder.
    """
    return Path(os.path.expanduser("~")) / ".config" / "probeinterface" / "library"


def download_probeinterface_file(manufacturer: str, probe_name: str, tag: Optional[str] = None) -> None:
    """Download the probeinterface file to the cache directory.
    Note that the file is itself a ProbeGroup but on the repo each file
    represents one probe.

    Parameters
    ----------
    manufacturer : "cambridgeneurotech" | "neuronexus" | "plexon" | "imec" | "sinaps"
        The probe manufacturer
    probe_name : str (see probeinterface_libary for options)
        The probe name
    tag : str | None, default: None
        Optional tag for the probe
    """
    cache_folder = get_cache_folder()
    if tag is not None:
        assert tag in get_tags_in_library(), f"Tag {tag} not found in library"
    else:
        tag = "main"

    os.makedirs(cache_folder / tag / manufacturer, exist_ok=True)
    local_file = cache_folder / tag / manufacturer / (probe_name + ".json")
    remote_file = public_url + tag + f"/{manufacturer}/{probe_name}/{probe_name}.json"
    rem = urlopen(remote_file)
    with open(local_file, "wb") as f:
        f.write(rem.read())


def get_from_cache(manufacturer: str, probe_name: str, tag: Optional[str] = None) -> Optional["Probe"]:
    """
    Get Probe from local cache

    Parameters
    ----------
    manufacturer : "cambridgeneurotech" | "neuronexus" | "plexon" | "imec" | "sinaps"
        The probe manufacturer
    probe_name : str (see probeinterface_libary for options)
        The probe name
    tag : str | None, default: None
        Optional tag for the probe

    Returns
    -------
    probe : Probe object, or None if no probeinterface JSON file is found

    """
    cache_folder = get_cache_folder()
    if tag is not None:
        cache_folder_tag = cache_folder / tag
        if not cache_folder_tag.is_dir():
            return None
        cache_folder = cache_folder_tag
    else:
        cache_folder_tag = cache_folder / "main"

    local_file = cache_folder_tag / manufacturer / (probe_name + ".json")
    if not local_file.is_file():
        return None
    else:
        probegroup = read_probeinterface(local_file)
        probe = probegroup.probes[0]
        probe._probe_group = None
        return probe


def get_probe(
    manufacturer: str,
    probe_name: str,
    name: Optional[str] = None,
    tag: Optional[str] = None,
    force_download: bool = False,
) -> "Probe":
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
    tag : str | None, default: None
        Optional tag for the probe
    force_download : bool, default: False
        If True, force re-download of the probe file.

    Returns
    ----------
    probe : Probe object

    """
    if not force_download:
        probe = get_from_cache(manufacturer, probe_name, tag=tag)
    else:
        probe = None

    if probe is None:
        download_probeinterface_file(manufacturer, probe_name, tag=tag)
        probe = get_from_cache(manufacturer, probe_name, tag=tag)
    if probe.manufacturer == "":
        probe.manufacturer = manufacturer
    if name is not None:
        probe.name = name

    return probe


def cache_full_library(tag=None) -> None:  # pragma: no cover
    """
    Download all probes from the library to the cache directory.
    """
    manufacturers = list_manufacturers(tag=tag)

    for manufacturer in manufacturers:
        probes = list_probes_by_manufacturer(manufacturer, tag=tag)
        for probe_name in probes:
            try:
                download_probeinterface_file(manufacturer, probe_name, tag=tag)
            except Exception as e:
                warnings.warn(f"Could not download {manufacturer}/{probe_name} (tag: {tag}): {e}")


def clear_cache(tag=None) -> None:  # pragma: no cover
    """
    Clear the cache folder for probeinterface library files.

    Parameters
    ----------
    tag : str | None, default: None
        Optional tag for the probe
    """
    cache_folder = get_cache_folder()
    if tag is not None:
        cache_folder_tag = cache_folder / tag
        if cache_folder_tag.is_dir():
            import shutil

            shutil.rmtree(cache_folder_tag)
    else:
        import shutil

        shutil.rmtree(cache_folder)


def list_manufacturers(tag=None) -> list[str]:
    """
    Get the list of available manufacturers in the library

    Returns
    -------
    manufacturers : list of str
        List of available manufacturers
    """
    if tag is not None:
        assert (
            tag in get_tags_in_library()
        ), f"Tag {tag} not found in library. Available tags are {get_tags_in_library()}."
    return list_github_folders("SpikeInterface", "probeinterface_library", ref=tag)


def list_probes_by_manufacturer(manufacturer: str, tag=None) -> list[str]:
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
    if tag is not None:
        assert (
            tag in get_tags_in_library()
        ), f"Tag {tag} not found in library. Available tags are {get_tags_in_library()}."
    assert manufacturer in list_manufacturers(
        tag=tag
    ), f"Manufacturer {manufacturer} not found in library. Available manufacturers are {list_manufacturers(tag=tag)}."
    return list_github_folders("SpikeInterface", "probeinterface_library", path=manufacturer, ref=tag)


def list_all_probes(tag=None) -> dict[str, list[str]]:
    """
    Get the list of all available probes in the library

    Returns
    -------
    all_probes : dict
        Dictionary with manufacturers as keys and list of probes as values
    """
    all_probes = {}
    manufacturers = list_manufacturers(tag=tag)
    for manufacturer in manufacturers:
        probes = list_probes_by_manufacturer(manufacturer, tag=tag)
        all_probes[manufacturer] = probes
    return all_probes


def get_tags_in_library() -> list[str]:
    """
    Get the list of available tags in the library

    Returns
    -------
    tags : list of str
        List of available tags
    """
    tags = get_all_tags("SpikeInterface", "probeinterface_library")
    return tags


### UTILS
def get_all_tags(owner: str, repo: str, token: str = None):
    """
    Get all tags for a repo.
    Returns a list of tag names, or an empty list if no tags exist.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/tags"
    headers = {}
    if token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN"):
        token = token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
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
    if token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN"):
        token = token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API returned status {resp.status_code}: {resp.text}")
    items = resp.json()
    return [
        item["name"]
        for item in items
        if item.get("type") == "dir" and item["name"][0] != "." and item["name"] != "scripts"
    ]
