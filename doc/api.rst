API
===



Probe
----------------

.. automodule:: probeinterface

    .. autoclass:: probeinterface.Probe
        :members:
    
ProbeBunch
----------------

.. automodule:: probeinterface
    
    .. autoclass:: probeinterface.ProbeBunch
        :members:

Import/export to formats
----------------------------------------

.. automodule:: probeinterface.io

    .. autofunction:: read_probeinterface

    .. autofunction:: write_probeinterface
    
    .. autofunction:: read_prb
    
    .. autofunction:: write_prb
    
    .. autofunction:: read_cvs
    
    .. autofunction:: write_cvs
    
    .. autofunction:: read_spikeglx
    
    .. autofunction:: read_mearec
    
    .. autofunction:: read_nwb


Probe generators
----------------------------------------

.. automodule:: probeinterface.generator

    .. autofunction:: generate_fake_probe
    
    .. autofunction:: generate_fake_probe_bunch
    
    .. autofunction:: generate_tetrode
    
    .. autofunction:: generate_multi_columns_probe
    
    .. autofunction:: generate_linear_probe
    
    .. autofunction:: 

Plotting
--------------------

.. automodule:: probeinterface.plotting

    .. autofunction:: plot_probe
    
    .. autofunction:: plot_probe_bunch

