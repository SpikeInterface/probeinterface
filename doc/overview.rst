Overview
=========

.. currentmodule:: probeinterface

Introduction
---------------

To record neural electrical signals, extracellular neural probes are inserted in the brain.
Neural probes are (usually) multi-channel arrays able to record from multiple electrodes simultaneously, spanning from
a few channels (e.g. tetrodes) to high-dense silicon probes (e.g. Neuropixels - with up to 384 recorded channels).

These probes (especially silicon probes) generally have a complex layout (or geometry) and can be connected to the
recording system in multiple ways (wiring). To connect a neural probe to a recording device (e.g. Open Ephys, Blackrock,
Ripple, Plexon, Intan, Multi-channel System) a headstage is used that is connected to the main recording device.

The complexity of the probe wiring and device wiring leads to the difficult task to directly link
the **physical electrodes on the probe**  and the **logical channel index on the device**.

Recent *spike sorting* (i.e. methods to extract single neurons' activity from the extracellular recordings) algorithms
mainly strongly rely on the probe geometry to exploit the spatial distribution of the electrodes and improve their
performance.

Therefore, there is a need to correctly handle probe geometry and the wiring to the device in an easy-to-use and
standardized way.

As an example, imagine you have:
   * a **Neuronexus A1x32-Poly2** probe
   * with the **intan RHD2132** headstage using the **omnetics 1315** connector
   * connected on the **port B of Open Ephys board**

What would be your final channel mapping?

Of course one can sit down in the lab and try to figure it out...
The goal of :code:`probeinterface` is to make this time-consuming and error-prone step easier and standardized.


Scope
---------

The scope of this project is handle one (or several) Probe with three simple python classes:

- :py:class:`Shank`
- :py:class:`Probe`
- :py:class:`ProbeGroup`.

These classes handle:
  * probe geometry (2D or 3D electrode layout)
  * probe shape (contour of the probe)
  * shape and size of the electrodes
  * probe wiring to the recording device
  * combination of several probes: global geometry + global wiring

This package also provide:
  * read/write to a common format (JSON based)
  * read/write function to existing format (PRB, NWB, CSV, MEArec, SpikeGLX, ...)
  * plotting routines
  * generator functions to create user-defined probes


Goal 1 
---------

This common interface could be used by several projects for spike sorting and electrophysiology analysis:

  * `SpikeInterface <https://github.com/SpikeInterface/spikeinterface>`_: integrate this into spikeextractors to handle channel location and wiring
  * `NEO <https://github.com/NeuralEnsemble/python-neo>`_: handle array_annotations for AnalogSignal
  * `SpikeForest <https://spikeforest.flatironinstitute.org/>`_: use this package for plotting probe activity
  * `Phy <https://github.com/cortex-lab/phy>`_: integrate for probe display
  * `SpiKING Circus <https://github.com/spyking-circus/spyking-circus>`_: handle probe with this package
  * `Kilosort <https://github.com/MouseLand/Kilosort>`_: handle probe with this package
  * `Tridesclous <https://github.com/tridesclous/tridesclous>`_: handle probe with this package
  * ...and more


Goal 2
---------

Implement and maintain a collection of widely used probes in Neuroscience, for example:

  * `Neuronexus <https://neuronexus.com/support/mapping-and-wiring/probe-mapping/>`_
  * `IMEC <https://www.imec-int.com/en/expertise/lifesciences/neural-probes>`_
  * `Cambridge Neurotech <https://www.cambridgeneurotech.com/neural-probes>`_


Existing projects
-------------------

:code:`probeinterface` is not the first attempt to build a library of available probes. Here is a list of available
resources:

  * `JRClust probe library <https://github.com/JaneliaSciComp/JRCLUST/tree/master/probes>`_ - Matlab format
  * `Klusta probe library <https://github.com/kwikteam/probes>`_ - PRB format
  * `SpyKING Circus probe library <https://github.com/spyking-circus/spyking-circus/tree/master/probes>`_ - PRB format

All of these projects only describe the electrode positions. Furthermore there is a strong ambiguity for users
between the **electrode index on the probe** and the **channel index on device**.
This could lead to a wrong interpretation of the wiring.

With :code:`probeinterface` we try to provide a unified framework for probe description, handling, and a comprehensive
probe library.

Acknowledgements
--------------------

The :code:`probeinterface` is strongly based on the  `MEAutility <https://github.com/alejoe91/MEAutility>`_ package,
written by `Alessio Buccino <https://alessiobuccino.com/>`_.

While the general idea of having an enhanced probe description is present, the :code:`MEAutility` package mainly focuses
on handling probes for modeling purposes, hence missing the wiring concept, and it can only handle a single probe at a
time.

Witth :code:`probeinterface` the focus is also to combinade several Probes and to handle complex wiring
for experimental description.
