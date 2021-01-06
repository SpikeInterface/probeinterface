Format specifications
=====================

Probe interface propose a simple format based on JSON.
The format is more or less a trivial serialisation into a python
dictionary which is jsonified. The dictionary, itself, map every 
attributes from the Probe class.

In fact, the format itself describe a ProbeGroup, so several probes.
So the format is able to describe a simple unique probe for the geometry
but also a full experimental setup with several probes and there wiring
to device.


Here a description field by field.

Lets image we want to describe this probe with:
  * 8 channels
  * 2 shanks

 
  
TODO: format spec step by step
