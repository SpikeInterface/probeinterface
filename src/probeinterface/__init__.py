import importlib.metadata

__version__ = importlib.metadata.version("probeinterface")


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
from .library import get_probe
from .wiring import get_available_pathways
