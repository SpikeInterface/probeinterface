# probeinterface
<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/probeinterface/">
    <img src="https://img.shields.io/pypi/v/probeinterface.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
    <a href="https://probeinterface.readthedocs.io/">
    <img src="https://readthedocs.org/projects/probeinterface/badge/?version=latest" alt="latest documentation" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/probeinterface/probeinterface/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/probeinterface.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://github.com/SpikeInterface/probeinterface/actions/workflows/full_tests.yml/badge.svg">
    <img src="https://github.com/SpikeInterface/probeinterface/actions/workflows/full_tests.yml/badge.svg" alt="CI build status" />
    </a>
  </td>
</tr>
<tr>
	<td>Codecov</td>
	<td>
		<a href="https://codecov.io/github/SpikeInterface/probeinterface">
		<img src="https://codecov.io/gh/SpikeInterface/probeinterface/branch/main/graphs/badge.svg" alt="codecov" />
		</a>
	</td>
</tr>
</table>


A Python package to handle the layout, geometry, and wiring of silicon probes for extracellular electrophysiology experiments.

Please [Star](https://github.com/SpikeInterface/probeinterface/stargazers) the project to support us and [Watch](https://github.com/SpikeInterface/probeinterface/subscription) to always stay up-to-date!


## Goals

ProbeInterface aims to provide a common framework to handle probe information across neuroscience experiments.

ProbeInterface is used by the [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) package to attach probe information to a recording object.
You can find detailed documentation in the [SpikeInterface documentation](https://spikeinterface.readthedocs.io/en/latest/modules/core.html#handling-probes).

In practice, ProbeInterface is a lightweight package to handle:

  * probe contact geometry (both 2D and 3D layouts)
  * probe shape (contour of the probe, shape of channel contacts, etc.)
  * probe wiring to a device (the physical layout often doesn't match the channel ordering)
  * combining several probes into a device, with probe groups and a global wiring
  * exporting probe data into a common JSON file format
  * loading existing probe geometry files (Cambridge Neurotech, NeuroNexus, etc.) from the [probeinterface_library](https://github.com/SpikeInterface/probeinterface_library)

In addition, ProbeInterface also offers the following features:

  * `matplotlib`-based plotting functions
  * loading/saving probes using common formats (PRB, CSV, NWB, ...)
  * correctly handling SI units (um, mm)


## Documentation

Detailed documentation of the latest PyPI release of ProbeInterface can be found [here](https://probeinterface.readthedocs.io/en/0.2.18).

Detailed documentation of the development version of ProbeInterface can be found [here](https://probeinterface.readthedocs.io/en/latest).

## How to install probeinterface

You can install the latest version of `probeinterface` version with pip:

```bash
pip install probeinterface
```

To get the latest updates, you can install `probeinterface` from source:

```bash
git clone https://github.com/SpikeInterface/probeinterface.git
cd probeinterface
pip install -e .
cd ..
```


## Citation

If you find ProbeInterface useful in your research, please cite:

```bibtex
@article{garcia2022probeinterface,
  title={ProbeInterface: a unified framework for probe handling in extracellular electrophysiology},
  author={Garcia, Samuel and Sprenger, Julia and Holtzman, Tahl and Buccino, Alessio P},
  journal={Frontiers in Neuroinformatics},
  volume={16},
  pages={823056},
  year={2022},
  publisher={Frontiers Media SA}
}
```
