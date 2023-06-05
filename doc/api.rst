API
===



Probe
-----

.. automodule:: probeinterface

    .. autoclass:: probeinterface.Probe
        :members:

ProbeGroup
----------

.. automodule:: probeinterface

    .. autoclass:: probeinterface.ProbeGroup
        :members:

Import/export to formats
------------------------

.. automodule:: probeinterface.io

    .. autofunction:: read_probeinterface

    .. autofunction:: write_probeinterface

    .. autofunction:: read_prb

    .. autofunction:: write_prb

    .. autofunction:: read_csv

    .. autofunction:: write_csv

    .. autofunction:: read_spikeglx

    .. autofunction:: read_mearec

    .. autofunction:: read_nwb


Probe generators
----------------

.. automodule:: probeinterface.generator

    .. autofunction:: generate_dummy_probe

    .. autofunction:: generate_dummy_probe_group

    .. autofunction:: generate_tetrode

    .. autofunction:: generate_multi_columns_probe

    .. autofunction:: generate_linear_probe

Plotting
--------

.. automodule:: probeinterface.plotting

    .. autofunction:: plot_probe

    .. autofunction:: plot_probe_group

Library
-------

.. automodule:: probeinterface.library

    .. autofunction:: get_probe
