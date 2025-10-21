Probeinterface public library
=============================

Probeinterface also handles a collection of probe descriptions in the
`ProbeInterface library <https://github.com/SpikeInterface/probeinterface_library>`_

The python module has a simple function to download and cache locally by using ``get_probe(...)``:


.. code-block:: python

    from probeinterface import get_probe
    probe = get_probe(
        manufacturer='neuronexus',
        probe_name='A1x32-Poly3-10mm-50-177'
    )


Once a probe is downloaded, it is cached locally for future use.

There are several helper functions to explore the library:

.. code-block:: python

    from probeinterface.library import (
        list_manufacturers,
        list_probes_by_manufacturer,
        list_all_probes
    )

    # List all manufacturers
    manufacturers = list_manufacturers()

    # List all probes for a given manufacturer
    probes = list_probes_by_manufacturer('neuronexus')

    # List all probes in the library
    all_probes = list_all_probes()

    # Cache all probes locally
    cache_full_library()


Each function has an optional ``tag`` argument to specify a git tag/branch/commit to get a specific version of the library.


How to contribute to the library
--------------------------------

Each probe in the library is represented by a JSON file and an image.
To contribute a new probe to the library, follow these steps:

  1. Generate the JSON file with probeinterface (or directly with another language)
  2. Generate an image of the probe with the `plot_probe` function in probeinterface
  3. Clone the `probeinterface_library repo <https://github.com/SpikeInterface/probeinterface_library>`_
  4. Put the JSON file and image into the correct folder: ``probeinterface_library/<manufacturer>/<model_name>/```
  5. Push to one of your branches with a git client
  6. Make a pull request to the main repo
