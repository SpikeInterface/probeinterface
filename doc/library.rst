Probeinterface public library
=============================

Probeinterface also handle a collection of probe description on the
`gin platform <https://gin.g-node.org/spikeinterface/probeinterface_library>`_

The python module have simple function to download and chache locally `get_probe(...)` ::

    from probeinterface import get_probe
    probe = get_probe(manufacturer='neuronexus',
                probe_name='A1x32-Poly3-10mm-50-177')

The gin platform is a github like platform that make possible to handle "big files" with git annex.
So user contribution in gin is as easy as a standard github pull request.

We expect to build rapidly commonly used probes in this public repository.

How to contribute
-----------------

TODO: exclain with more details

  1. Genertae the JSON file with probeinterface function (or directly
      with another language)
  2. Generate animage with `plot_probe`
  3. Clone with gin client the `probeinterface_library repo <https://gin.g-node.org/spikeinterface/probeinterface_library>`_
  4. Put files at the good place.
  5. Ask for an account
  6. Push to a branch with gin client
  7. Make a pull request on the gin portal (like a github PR)



