Probeinterface public library
=============================

Probeinterface also handles a collection of probe descriptions on the
`GitHub platform <https://github.com/SpikeInterface/probeinterface_library>`_

The python module has a simple function to download and cache locally by using `get_probe(...)` ::

    from probeinterface import get_probe
    probe = get_probe(manufacturer='neuronexus',
                probe_name='A1x32-Poly3-10mm-50-177')


We expect to build rapidly commonly used probes in this public repository.

How to contribute
-----------------

TODO: explain with more details

  1. Generate the JSON file with probeinterface function (or directly
      with another language)
  2. Generate an image with `plot_probe`
  3. Clone with gin client the `probeinterface_library repo <https://github.com/SpikeInterface/probeinterface_library>`_
  4. Put files in the right place.
  5. Push to a branch with git client
  6. Make a pull request on the gin portal (like a github PR)
