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
  * 2 shanks (one tetrode on each shank)

.. image:: img/probe_format_example.png
   :width: 800 px





The first part contain field that give the version of probeinterface
and a list of probes::

  {
    "specification": "probeinterface",
    "version": "0.1.0",
    "probes": [
      {
        ...
      }
    ]
  }

Then each probe will be a sub dictionary in the probes list::

        {
            "ndim": 2,
            "si_units": "um",
            "annotations": {
                "name": "2 shank tetrodes",
                "manufacturer": "homemade"
            },
            "electrode_positions": [
        ...

This dict contain neceassy fields and optional fields.

Necessary:
  * ndim
  * si_units
  * annotations
  * electrode_positions
  * electrode_shapes
  * electrode_shape_params

Optional:
  * electrode_plane_axes
  * probe_planar_contour
  * device_channel_indices
  * shank_ids


Here the full format

.. include:: probe_format_example.json
    :code: json
