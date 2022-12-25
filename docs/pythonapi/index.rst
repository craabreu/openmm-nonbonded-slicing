Python API
==========

To use the plugin from your Python script:

.. code-block:: python

    import openmm as mm
    import nonbondedslicing as plugin
    system = mm.System()
    force = plugin.SlicedPmeForce(1)
    system.addForce(force)

Implemented subclass of :OpenMM:`Force`:

.. toctree::
    :glob:

    SlicedPmeForce
    SlicedNonbondedForce


.. testsetup::

    from nonbondedslicing import *
