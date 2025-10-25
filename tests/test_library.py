import os
import pytest

from probeinterface import Probe
from probeinterface.library import (
    download_probeinterface_file,
    get_from_cache,
    get_probe,
    get_tags_in_library,
    list_manufacturers,
    list_probes_by_manufacturer,
    list_all_probes,
    get_cache_folder,
    cache_full_library,
    clear_cache,
)


manufacturer = "neuronexus"
probe_name = "A1x32-Poly3-10mm-50-177"


def _remove_from_cache(manufacturer: str, probe_name: str, tag=None) -> None:
    """
    Remove Probe from local cache

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
        cache_folder_tag = cache_folder / tag
        if not cache_folder_tag.is_dir():
            return None
        cache_folder = cache_folder_tag
    else:
        cache_folder_tag = cache_folder / "main"

    local_file = cache_folder_tag / manufacturer / (probe_name + ".json")
    if local_file.is_file():
        os.remove(local_file)


def test_download_probeinterface_file():
    download_probeinterface_file(manufacturer, probe_name, tag=None)


def test_get_from_cache():
    download_probeinterface_file(manufacturer, probe_name)
    probe = get_from_cache(manufacturer, probe_name)
    assert isinstance(probe, Probe)

    tag = get_tags_in_library()[0]
    probe = get_from_cache(manufacturer, probe_name, tag=tag)
    assert probe is None  # because we did not download with this tag
    download_probeinterface_file(manufacturer, probe_name, tag=tag)
    probe = get_from_cache(manufacturer, probe_name, tag=tag)
    _remove_from_cache(manufacturer, probe_name, tag=tag)
    assert isinstance(probe, Probe)

    probe = get_from_cache("yep", "yop")
    assert probe is None


def test_get_probe():
    probe = get_probe(manufacturer, probe_name)
    assert isinstance(probe, Probe)
    assert probe.get_contact_count() == 32


def test_available_tags():
    tags = get_tags_in_library()
    if len(tags) > 0:
        for tag in tags:
            assert isinstance(tag, str)
            assert len(tag) > 0


@pytest.mark.library
def test_list_manufacturers():
    manufacturers = list_manufacturers()
    assert isinstance(manufacturers, list)
    assert "neuronexus" in manufacturers
    assert "imec" in manufacturers


@pytest.mark.library
def test_list_probes():
    manufacturers = list_all_probes()
    for manufacturer in manufacturers:
        probes = list_probes_by_manufacturer(manufacturer)
        assert isinstance(probes, list)
        assert len(probes) > 0


@pytest.mark.skip(reason="long test that downloads the full library")
def test_cache_full_library():
    tag = get_tags_in_library()[0] if len(get_tags_in_library()) > 0 else None
    print(tag)
    cache_full_library(tag=tag)
    all_probes = list_all_probes(tag=tag)
    # spot check that a known probe is in the cache
    for manufacturer, probes in all_probes.items():
        for probe_name in probes:
            probe = get_from_cache(manufacturer, probe_name, tag=tag)
            assert isinstance(probe, Probe)

    clear_cache(tag=tag)
    for manufacturer, probes in all_probes.items():
        for probe_name in probes:
            probe = get_from_cache(manufacturer, probe_name, tag=tag)
            assert probe is None


if __name__ == "__main__":
    test_download_probeinterface_file()
    test_get_from_cache()
    test_get_probe()
    test_list_manufacturers()
    test_list_probes()
    test_cache_full_library()
