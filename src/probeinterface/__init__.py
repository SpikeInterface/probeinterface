import importlib.metadata
import importlib.util
from packaging.version import parse

__version__ = importlib.metadata.version("probeinterface")

# If Zarr is installed, it must be >= 3.0.0
ZARR_INSTALLED = importlib.util.find_spec("zarr") is not None
if ZARR_INSTALLED:
    import zarr

    if parse(zarr.__version__) < parse("3.0.0"):
        raise ImportError("zarr version must be >= 3.0.0")


from .probe import Probe, select_axes
from .probegroup import ProbeGroup
from .io import (
    write_probeinterface,
    read_probeinterface,
    read_prb,
    write_prb,
    read_csv,
    write_csv,
    read_BIDS_probe,
    write_BIDS_probe,
    read_spikegadgets,
    read_mearec,
    read_nwb,
    read_maxwell,
    read_3brain,
)
from .neuropixels_tools import (
    read_imro,
    write_imro,
    read_spikeglx,
    parse_spikeglx_meta,
    parse_spikeglx_snsGeomMap,
    get_saved_channel_indices_from_spikeglx_meta,
    read_openephys,
    get_saved_channel_indices_from_openephys_settings,
)
from .utils import combine_probes
from .generator import (
    generate_dummy_probe,
    generate_dummy_probe_group,
    generate_tetrode,
    generate_linear_probe,
    generate_multi_columns_probe,
    generate_multi_shank,
)
from .library import (
    get_probe,
    list_manufacturers,
    list_probes_by_manufacturer,
    list_all_probes,
    get_tags_in_library,
    cache_full_library,
    clear_cache,
)
from .wiring import get_available_pathways
