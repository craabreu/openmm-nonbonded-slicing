Python API
==========

To use the plugin from your Python script, you can do:

.. code-block:: python

    import openmm as mm
    import nonbondedslicing as nbs
    system = mm.System()
    force = nbs.SlicedNonbondedForce(2)
    system.addForce(force)

This is the implemented subclass of :OpenMM:`Force`:

.. toctree::
    :titlesonly:

    SlicedNonbondedForce


.. testsetup::

    from nonbondedslicing import *
