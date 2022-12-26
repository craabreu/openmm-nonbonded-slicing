Python API
==========

To use the plugin from your Python script:

.. code-block:: python

    import openmm as mm
    import nonbondedslicing as plugin
    system = mm.System()
    force = plugin.SlicedNonbondedForce(1)
    system.addForce(force)

Implemented subclass of :OpenMM:`Force`:

.. toctree::
    :glob:

    SlicedNonbondedForce


.. testsetup::

    from nonbondedslicing import *
