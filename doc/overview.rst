Overview
=======

.. currentmodule:: probeinterface

Preample
---------------

To record neural electrical signals, manufacturers provides "silicon probe".
Theses are multi channel electrodes arrays able to record from four to thousands channels  at once.

Theses probes have generally a complex mapping (aka layout, aka geometry) and also complex wiring.

Theses probes are themself connect to a recording device  (openephys, blackrock, ripple, plexon, itan, multichannelsystem...).
The device itself is generally a headstage (small amplifier) and the main device (headstage hub).

The complexity of the probe wiring and device wiring lead to very difficult task to make link
between **physical electrodeon the probe**  and **logical channel index on the device** .

Recent spike sorting algorithm rely mainly on the probe geometry (aka mapping).
So there is a need to handle correctly probe geometry and the underlying wiring.


One example , imagine you have :
   * A probe **neuronexus A1x32-Poly2**
   * with the headstage **intan RHD32** using **omnetics 1315**
   * connected on the **port B of open ephys board**
What would be the final channel mapping ?
This is a total headache for the end user.
Anyone having done it once know it  totally.


Scope
---------

The scope of this project is handle one (or several) Probe with two simple python classes : :py:class:`Probe` and 
:py:class:`ProbeBunch`.

Theses class handle: 
  * probe geometry (2d, 3d electrode layout)
  * probe shape (contour of the probe)
  * shape of electrodes
  * probe wiring to device (channel are not in order generaly)
  * combinaison of several probe : global geometry + global wiring


This package also provide:
  * read/write to a NEW format (hdf5 based)
  * read/write function to existing format (PRB, CSV, spikeglx, mearec, ...)
  * plotting example
  * generator of simple shape


Goal 1 
---------

This common interface could be used by several projects for spike sorting and ephy analysis:

  * spikeinterface team : integrate this into spikeextractor for channel location
  * neo team : handle array_annotations for AnalogSignal
  * spikeforest team : use this package for ploting probe activity
  * phy team: integrate for probe display
  * spyking-circus team : handle probe with this package
  * kilosort team : handle probe with this package
  * tridesclous team : handle probe with this package
  * ...


Goal 2
---------

If this package is widely adopted, then I plan to implement a collections of widly use probe layout :

  * `neuronexus <https://neuronexus.com/support/mapping-and-wiring/probe-mapping/>`_
  * `imec <https://www.imec-int.com/en/expertise/lifesciences/neural-probes>`_
  * `cambridge neurotech <https://www.cambridgeneurotech.com/neural-probes>`_


Already existing  projects
-------------------------------------

prointerface is not the first tentative of doing this:

  * The JRclust team already start a collection of probe descrition with matlab `here <https://github.com/JaneliaSciComp/JRCLUST/tree/master/probes>`_
  * The klusta team already start a collection of probe `here <https://github.com/kwikteam/probes>`_ with PRB format.
  * The spyking circus team also did something `similar <https://github.com/spyking-circus/spyking-circus/tree/master/probes>`_ with PRB format also

All of theses projects describe only the electrode positions. Furthermore there is a strong amibiguity for users
in between **electrode index on probe** and **channel index on device**.
So if one probe is pluged in another port on the device then the wiring is wrong!
  
Here, in probeinterface we package try a deeper description with multi probe description,
shape handling, 3d, device indices, main axes, ...
  

Aknowledgement
---------------------------

This work is based on the package `MEAutility <https://github.com/alejoe91/MEAutility>`_ made by Alessio buccino.

The MEAutility is focusing on generating current over electrodes for MEArec.

Here, in probeinterface the focus is to combinate several Probe to handle complex wiring
for experimental description.
