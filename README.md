# probeinterface

A Python package to handle the layout, geometry, and wiring of silicon probes for extracellular electrophysiology experiments.

Documentation : https://probeinterface.readthedocs.io/


## Goals

Make a lightweight package to handle:

  * probe contact geometry (both 2D and 3D layouts)
  * probe shape (contour of the probe, shape of channel contact, ...)
  * probe wiring to device (the physical layout often doesn't match the channel ordering)
  * combining several probes into a device with global geometry + global wiring
  * exporting probe geometry data into JSON files
  * loading existing probe geometry files (Neuronexus, imec, Cambridge Neurotech...) [Started here](https://gin.g-node.org/spikeinterface/probeinterface_library)

Bonus :

  * optional plotting (based on `matplotlib`)
  * load/save geometry using common formats (PRB, CSV, NWB, ...)
  * handle SI length units correctly um/mm/...


Target users/projet :

  * spikeinterface team : integrate this into spikeextractor for channel location
  * neo team : handle array_annotations for AnalogSignal
  * spikeforest team : use this package for ploting probe activity
  * phy team: integrate for probe display
  * spyking-circus team : handle probe with this package
  * kilosort team : handle probe with this package
  * tridesclous team : handle probe with this package
  * open ephys team : automatically generate channel map configuration files


 Author: Samuel Garcia

