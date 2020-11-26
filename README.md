# probeinterface

Python package to handle probe layout, geometry and wiring to device.


## Goal

Make a ligthweigted package usefull that handle:

  * probe geometry (2d, 3d electrode layout)
  * probe shape (shape of the probe, shape of channel contact, ...)
  * probe wiring to device (channel are not in order generaly)
  * combinaison of several probe : global geometry + global wiring

Bonus (maybe):

  * handle a collection of existing probe (neuronexus, imec, ...)
  * optional ploting with matplotlib include
  * load/save from/into several possible formats
  * handle SI correctly um/mm/...

  
Target users/projet :

  * spikeinterface team : integrate this into spikeextractor for channel location
  * neo team : handle array_annotations for AnalogSignal
  * spikeforest team : use this package for ploting probe activity
  * phy team: integrate for probe display
  * spyking-circus team : handle probe with this package
  * kilosort team : handle probe with this package
  * tridesclous team : handle probe with this package
 
  
Constrain:

  * be compatible with existing (PRB format, csv format, NWB format , .mat KS2 format...)
