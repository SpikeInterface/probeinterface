Doc probeinterface
===============

Python package to handle probe layout, geometry and wiring to device. 


The package handle mainly:

  * probe geometry (2d, 3d electrode layout)
  * probe shape (contour of the probe)
  * shape of electrodes
  * probe wiring to device (channel are not in order generaly)
  * combinaison of several probe : global geometry + global wiring

The probeinterface package also provide:

  * basic plotting function with matplotlib (to demonstrate the use)
  * input/ouput to several formats (PRB, CSV, mearec, spikeglx, ...)
  





.. include:: examples/index.rst

   
.. toctree::
   :caption: Contents:
   :maxdepth: 2
   
   overview
   api
   examples/index.rst




..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
