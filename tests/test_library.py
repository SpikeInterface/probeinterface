from probeinterface import Probe
from probeinterface.library import (
    download_probeinterface_file,
    get_from_cache,
    get_probe,
    get_tags_in_library,
    get_manufacturers_in_library,
    get_probes_in_library,
)


from pathlib import Path
import numpy as np

import pytest


manufacturer = "neuronexus"
probe_name = "A1x32-Poly3-10mm-50-177"


def test_download_probeinterface_file():
    download_probeinterface_file(manufacturer, probe_name)


def test_get_from_cache():
    download_probeinterface_file(manufacturer, probe_name)
    probe = get_from_cache(manufacturer, probe_name)
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


def test_get_manufacturers_in_library():
    manufacturers = get_manufacturers_in_library()
    assert isinstance(manufacturers, list)
    assert "neuronexus" in manufacturers
    assert "imec" in manufacturers


def test_get_probes_in_library():
    manufacturers = get_manufacturers_in_library()
    for manufacturer in manufacturers:
        probes = get_probes_in_library(manufacturer)
        assert isinstance(probes, list)
        assert len(probes) > 0


if __name__ == "__main__":
    test_download_probeinterface_file()
    test_get_from_cache()
    test_get_probe()
    test_get_latest_tag()
    test_get_manufacturers_in_library()
    test_get_probes_in_library()
