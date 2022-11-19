=========================
OpenMM PME Slicing Plugin
=========================

.. image:: https://badgen.net/badge/icon/github?icon=github&label
   :target: https://github.com/craabreu/openmm-pme-slicing

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://github.com/craabreu/openmm-pme-slicing/blob/main/LICENSE.md

This `OpenMM <https://openmm.org>`_ plugin implements a sliced variant of the smooth Particle Mesh Ewald (PME) method.
By partitioning all particles among :math:`n` non-interesecting subsets, the total Coulomb potential
becomes

.. math::

    U = \sum_{I=0}^{n-1} \sum_{J=I}^{n-1} h_{I,J} U_{I,J}

where :math:`h_{I,J}` is a switching constant and :math:`U_{I,J}` is the sum over every pair of a
particle in subset I and another particle in subset J.

========
Contents
========

.. toctree::
   :maxdepth: 2

   overview
   pythonapi/index
   contributing
   authors
   changelog
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
