The Neuropixels catalogue pattern
=================================

.. currentmodule:: probeinterface


The catalogue: :py:func:`build_neuropixels_probe`
-------------------------------------------------

The foundation of every Neuropixels reader in probeinterface is
:py:func:`build_neuropixels_probe`. Given a probe part number (a specific
stock-keeping unit identifier such as ``"NP1000"``, ``"NP2000"``, ``"NP2014"``),
it returns a :py:class:`Probe` carrying the full silicon geometry for that
part number: every catalogue contact (960 for Neuropixels 1.0, 1280 per shank
for Neuropixels 2.0), the planar contour of the shanks, the contact shapes and
sizes, the analog-to-digital converter (ADC) multiplexer (MUX) routing table,
and the probe-level annotations (``manufacturer``, ``model_name``,
``part_number``, ``description``).

The numbers behind that geometry come from the
`ProbeTable <https://github.com/billkarsh/ProbeTable>`_ repository maintained
by `Bill Karsh <https://github.com/billkarsh>`_ (author of SpikeGLX).
ProbeTable is the canonical machine-readable inventory of IMEC Neuropixels
probe specifications: contact positions, electrode dimensions, shank geometry,
MUX routing, ADC configuration, all keyed by part number. Probeinterface
mirrors a postprocessed snapshot of that data into the package via
``resources/postprocess_neuropixels_probe_features.py``, which is re-run after
each ProbeTable sync. Without ProbeTable, every reader would have to carry
its own hand-written copy of the manufacturer specs, which is exactly the
situation the catalogue pattern is designed to avoid.


The format readers
------------------

Four entry points read Neuropixels recordings (or recording configurations)
and produce a probe ready to use with SpikeInterface:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Reader
     - Input
     - Where the part number comes from
   * - :py:func:`read_spikeglx`
     - SpikeGLX ``.ap.meta`` (plus the ``.ap.bin`` it describes)
     - ``imDatPrb_pn`` field in the meta file
   * - :py:func:`read_openephys_neuropixels`
     - Open Ephys ``settings.xml`` (plus the binary stream it describes)
     - ``probe_part_number`` attribute in the XML
   * - :py:func:`read_imro`
     - SpikeGLX IMRO (Imec ReadOut) table file (``.imro``)
     - First field of the IMRO header: a part number directly (new SpikeGLX
       format) or a legacy numeric type code translated to a part number via
       the catalogue mapping (old format). See SpikeGLX issue
       `#432 <https://github.com/SpikeInterface/probeinterface/issues/432>`_
       for the format transition.
   * - :py:func:`read_spikegadgets_neuropixels`
     - SpikeGadgets ``.rec`` XML header
     - Not present in the file; the reader picks a geometry-equivalent stand-in
       based on ``(SpikeConfiguration.device, deviceSubType)``: ``NP1000`` for
       Neuropixels 1.0, ``NP2000`` for Neuropixels 2.0 single-shank, ``NP2014``
       for Neuropixels 2.0 4-shank

The first three readers read the part number directly from the recording
metadata. SpikeGadgets is the exception: its ``.rec`` XML does not carry a
part number field, so the reader cannot know which specific variant produced
the recording. It picks one representative per geometry-equivalent family
(all Neuropixels 1.0 staggered variants share contact positions; all
Neuropixels 2.0 single-shank variants share contact positions; all
Neuropixels 2.0 4-shank variants share contact positions) and clears the
``model_name``, ``description``, and ``part_number`` annotations on the
returned probe so downstream code does not read the stand-in as an
attribution.


From catalogue probe to probe in a recording setup
--------------------------------------------------

The catalogue probe is pure geometry, divorced from any recording session. A real
recording uses only a subset of those contacts: the Neuropixels headstage
acquires 384 channels at a time, and the recording configuration selects
which catalogue contacts those 384 are drawn from (384 of 960 on Neuropixels
1.0, 384 of 1280 per shank on Neuropixels 2.0 single-shank, 384 of 5120 on
Neuropixels 2.0 4-shank). The selection mechanism differs by recording
format (an IMRO table for SpikeGLX, a channel map in ``settings.xml`` for
Open Ephys, the ``SpikeNTrode`` list in SpikeGadgets's ``.rec`` XML); each
reader's docstring covers the specifics for that format. On top of the
selection, the recording adds session-specific state:
per-contact analog band (AP) and local field potential (LFP) gains, ADC
sample order, reference configuration, and the channel-to-file mapping that
says where each contact's data lives in the saved binary. Probeinterface
calls the result a probe in a recording setup, to distinguish it from the
catalogue.

Each reader produces the recording-setup probe in the same three steps:

1. Build the catalogue probe by calling
   :py:func:`build_neuropixels_probe(part_number) <build_neuropixels_probe>`
   with the part number obtained from the recording metadata.
2. Slice the catalogue probe to the active electrodes for this recording session via
   :py:meth:`probe.get_slice(active_indices) <Probe.get_slice>`. The slice
   drops the unrecorded contacts but preserves the probe-level annotations
   and the per-contact catalogue annotations (ADC group, sample order) on the
   contacts that survive.
3. Attach the recording-specific state: per-contact AP/LFP gains, any
   reference annotations, and finally
   :py:meth:`probe.set_device_channel_indices(...) <Probe.set_device_channel_indices>`
   to record where each surviving contact's data lives in the saved file.


What the pattern solves
-----------------------

Constructing geometry from scratch inside each format reader (the situation
before the catalogue pattern) had three problems:

* **Geometry drift across readers.** Each reader carried its own copy of the
  manufacturer specs. A Neuropixels 2.0 4-shank probe loaded through SpikeGLX
  and through SpikeGadgets could return contact positions that disagreed in
  the third decimal because the two readers had been updated against
  different snapshots of the IMEC spec. Centralising the geometry in
  :py:func:`build_neuropixels_probe` and sourcing it from ProbeTable means
  every reader returns the same positions for the same part number.
* **Conflated geometry and wiring bugs.** When a saved recording looked wrong
  on the probe, it was difficult to say whether the geometry was off
  (catalogue issue) or the channel-to-contact mapping was off (wiring issue).
  With the two phases separated, a geometry bug is a bug in
  :py:func:`build_neuropixels_probe`; a wiring bug is a bug in the reader's
  matching step. The two can be diagnosed and fixed independently.
* **Hidden active-electrode selection.** Readers that built a 384-contact
  probe directly hid the fact that 576 catalogue contacts were silently
  dropped. The explicit ``probe.get_slice(active_indices)`` step makes the
  selection visible and inspectable: callers can ask "which catalogue
  contacts did this recording session record?" and get a direct answer.

The pattern also pays out on the upgrade path. When IMEC ships a new probe
variant, the integration work is "add the part number to ProbeTable, re-run
the postprocess script".


Discussion
----------

This pattern was proposed and is tracked in issue
`#405 <https://github.com/SpikeInterface/probeinterface/issues/405>`_; if you have
any discussion point to add please re-open the issue so the maintainers can discuss.
