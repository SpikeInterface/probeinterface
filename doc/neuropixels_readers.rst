The Neuropixels catalogue pattern
=================================

.. currentmodule:: probeinterface


Two kinds of Neuropixels probe
------------------------------

Probeinterface distinguishes two kinds of Neuropixels probe.

**Catalogue probe.** The probe as it appears in the IMEC catalogue, with
every contact on the silicon present (960 on Neuropixels 1.0, 1280 per
shank on Neuropixels 2.0). Built via :py:func:`build_neuropixels_probe(part_number)
<build_neuropixels_probe>`. It carries:

* contact positions
* shank contour and dimensions
* contact shapes and sizes
* the analog-to-digital converter (ADC) multiplexer (MUX) routing on the
  silicon
* identity metadata (manufacturer, model name, part number, description)

A catalogue probe is pure geometry, the same for every recording made with
that variant. Use it to plot the probe layout, compute distances between
contacts, or run any analysis that does not depend on a specific recording.

**Recording-setup probe.** The catalogue probe specialised for one
recording session: only the contacts actually recorded are present
(typically 384 of the 960 or more catalogue contacts), and per-contact
recording state is attached:

* per-contact analog band (AP) and local field potential (LFP) gains
* reference configuration
* per-contact sampling order
* probe wiring (the mapping from each contact to the recording channel
  that captured its data)

Use a recording-setup probe to hand the recording to SpikeInterface so
spike sorters see both the geometry and the correct channel mapping, to
convert raw samples to microvolts via the per-contact gains, or to plot
the recorded contacts alongside the recorded traces.


The catalogue source
--------------------

A probe part number (SKU) is an IMEC identifier such as ``"NP1000"``,
``"NP2000"``, or ``"NP2014"``. The part number determines the silicon
geometry: 960 contacts on Neuropixels 1.0, 1280 per shank on Neuropixels
2.0, plus all the per-variant pitch and shank dimensions.

The data behind the catalogue comes from the
`ProbeTable <https://github.com/billkarsh/ProbeTable>`_ repository maintained
by `Bill Karsh <https://github.com/billkarsh>`_ (author of SpikeGLX).
ProbeTable is the canonical machine-readable inventory of IMEC Neuropixels
probe specifications, all keyed by part number.


Format readers
--------------

A format reader turns a Neuropixels recording into a recording-setup probe
in three steps:

1. **Fetch the catalogue probe.** Look up the probe part number (SKU) in
   the recording's metadata, then call
   :py:func:`build_neuropixels_probe(part_number) <build_neuropixels_probe>`.
2. **Identify the active electrodes.** Read the recording's channel
   configuration to find which catalogue contacts were actually recorded,
   and slice the catalogue probe down to that subset via
   :py:meth:`probe.get_slice(active_indices) <Probe.get_slice>`.
3. **Attach the recording-setup metadata.** Add the per-contact gains,
   reference settings, and sampling order, and set the probe wiring.

The four readers in probeinterface differ in step 1: where they look up the
part number in the recording's metadata.

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
     - Not present in the file; the reader picks a geometry-equivalent
       stand-in based on ``(SpikeConfiguration.device, deviceSubType)``:
       ``NP1000`` for Neuropixels 1.0, ``NP2000`` for Neuropixels 2.0
       single-shank, ``NP2014`` for Neuropixels 2.0 4-shank

The first three readers read the part number directly. SpikeGadgets is the
exception (its ``.rec`` XML does not carry a part number) and falls back to
a geometry-equivalent stand-in; the variants within each Neuropixels family
share identical 2D contact geometry, so any representative produces correct
positions.


What the pattern solves
-----------------------

Building geometry from scratch inside each reader (the situation before this
pattern) caused three problems:

* **Drift across readers.** Each reader carried its own copy of the
  manufacturer specs, and the copies could disagree. Centralising the
  geometry in :py:func:`build_neuropixels_probe` and sourcing it from
  ProbeTable means every reader returns the same positions for the same
  part number.
* **Confused bugs.** When a saved recording looked wrong on the probe, it
  was hard to tell whether the geometry was off or the channel-to-contact
  mapping was off. With the two phases separated, a geometry bug is a bug
  in :py:func:`build_neuropixels_probe`; a wiring bug is a bug in the
  reader's slicing or wiring step. The two can be diagnosed independently.
* **Hidden electrode selection.** A reader that built a 384-contact probe
  directly hid the fact that 576 catalogue contacts were silently dropped.
  The explicit slice step makes the selection visible: callers can ask
  which catalogue contacts the recording captured and get a direct answer.

The pattern also helps on the upgrade path. When IMEC ships a new probe
variant, the integration work is to add the part number to ProbeTable; the
readers do not change.


Discussion
----------

This pattern was proposed and is tracked in issue
`#405 <https://github.com/SpikeInterface/probeinterface/issues/405>`_.
Reopen the issue if you want to discuss changes.
