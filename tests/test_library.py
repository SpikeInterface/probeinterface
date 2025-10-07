from probeinterface import Probe
from probeinterface.library import (
    download_probeinterface_file,
    get_from_cache,
    remove_from_cache,
    get_probe,
    get_tags_in_library,
    list_manufacturers_in_library,
    list_probes_in_library,
    get_cache_folder,
)


manufacturer = "neuronexus"
probe_name = "A1x32-Poly3-10mm-50-177"


def test_download_probeinterface_file():
    download_probeinterface_file(manufacturer, probe_name, tag=None)


def test_latest_commit_mechanism():
    _ = get_probe(manufacturer, probe_name)
    cache_folder = get_cache_folder()
    latest_commit_file = cache_folder / "main" / "latest_commit.txt"
    assert latest_commit_file.is_file()

    # now we manually change latest_commit.txt to something else
    with open(latest_commit_file, "w") as f:
        f.write("1234567890123456789012345678901234567890")

    # now we get the probe again and make sure the latest_commit.txt file is updated
    _ = get_probe(manufacturer, probe_name)
    assert latest_commit_file.is_file()
    with open(latest_commit_file, "r") as f:
        latest_commit = f.read().strip()
    assert latest_commit != "123456789012345678901234567890123456789"


def test_get_from_cache():
    download_probeinterface_file(manufacturer, probe_name)
    probe = get_from_cache(manufacturer, probe_name)
    assert isinstance(probe, Probe)

    tag = get_tags_in_library()[0]
    probe = get_from_cache(manufacturer, probe_name, tag=tag)
    assert probe is None  # because we did not download with this tag
    download_probeinterface_file(manufacturer, probe_name, tag=tag)
    probe = get_from_cache(manufacturer, probe_name, tag=tag)
    remove_from_cache(manufacturer, probe_name, tag=tag)
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


def test_list_manufacturers_in_library():
    manufacturers = list_manufacturers_in_library()
    assert isinstance(manufacturers, list)
    assert "neuronexus" in manufacturers
    assert "imec" in manufacturers


def test_list_probes_in_library():
    manufacturers = list_manufacturers_in_library()
    for manufacturer in manufacturers:
        probes = list_probes_in_library(manufacturer)
        assert isinstance(probes, list)
        assert len(probes) > 0


if __name__ == "__main__":
    test_download_probeinterface_file()
    test_get_from_cache()
    test_get_probe()
    test_list_manufacturers_in_library()
    test_list_probes_in_library()
