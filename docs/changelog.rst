=========
Changelog
=========

0.5.0 (unreleased)
------------------

**Breaking changes**

* With Ewald, PME or LJPME, the terms that subtract the reciprocal-space interactions of excluded
  pairs and exceptions are computed in the reciprocal-space force group, not in the direct-space
  one. They only cancel part of the reciprocal-space sum, so both must be evaluated together, as
  in multiple time stepping (Procacci et al., J. Chem. Phys. 108, 8799, 1998; Morrone et al.,
  J. Chem. Theory Comput. 6, 1798, 2010). This intentionally differs from OpenMM's
  ``NonbondedForce``. Total forces and energies are unchanged. These terms are also kept when
  ``setIncludeDirectSpace(False)`` is used.

0.4.0
-----

**Breaking changes**

* Energy parameter derivatives are only computed along with the energy, as in
  ``getState(getEnergy=True, getParameterDerivatives=True)``. Requested without the energy, they
  are NaN on all platforms. ``deriv(energy, ...)`` in a ``CustomIntegrator`` is not affected.

**Performance**

* GPU platforms: requested derivatives no longer slow down steps that compute only forces, such as
  molecular dynamics steps, and computing many derivatives is faster.

**Documentation**

* Measured costs of slicing and of derivatives in the theory section.
* Definition of the derivatives with respect to scaling parameters in the overview.

0.3.0
-----

Requires OpenMM 8.4 or later. The Python wrapper must be built with the same SWIG version as OpenMM.

**Breaking changes**

* ``addScalingParameterDerivative``, ``getNumScalingParameterDerivatives`` and
  ``getScalingParameterDerivativeName`` were renamed to ``addEnergyParameterDerivative``,
  ``getNumEnergyParameterDerivatives`` and ``getEnergyParameterDerivativeName``, as in OpenMM.
  ``setScalingParameterDerivative`` was removed.
* ``SlicedNonbondedForce.h`` no longer brings the ``std`` and ``OpenMM`` namespaces into code
  that includes it.
* ``nonbondedslicing.unit``, an unused re-export of ``openmm.unit``, was removed.

**New features**

* CUDA and OpenCL platforms rebuilt on OpenMM's common compute architecture.
* The CUDA platform can use either cuFFT or VkFFT (``setUseCuFFT``).
* Background (neutralizing plasma) energy of Ewald and PME, sliced among subsets.

**Bug fixes**

* Wrong energies and derivatives for systems with particle or exception parameter offsets.
* GPU platforms: ported OpenMM 8.4's fixes to the plasma correction with parameter offsets.
* GPU platforms: with parameter offsets and Ewald, PME or LJPME, derivatives of diagonal slices
  included the self energy when the direct-space force group was evaluated alone.
* ``setScalingParameter`` did not detect some clashes with other scaling parameters.
* ``useCuFFT`` was not serialized.
* Python: ``getScalingParameter`` could not be called, and ``copy``, ``deepcopy`` and ``pickle``
  returned unusable objects.

**Build and packaging**

* CMake honors ``OPENMM_DIR``, ``OPENMM_VERSION`` and ``VKFFT_INCLUDE_DIR``, and finds CUDA
  with ``FindCUDAToolkit``.
* The Python extension is built with setuptools.

**Tests**

* Every slice energy and derivative is checked against ``NonbondedForce`` on all platforms.
* Tests for 3 subsets, parameter offsets, net charge, API errors, serialization and the Python
  wrapper; platform-specific CUDA and OpenCL tests run again.
