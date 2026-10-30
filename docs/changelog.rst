=========
Changelog
=========

0.5.0 (unreleased)
------------------

**Breaking changes**

* ``getScalingParameter`` also returns the ``includeCoulombDecapping`` and ``includeLJDecapping``
  flags: two more output arguments in C++, and a list of seven items instead of five in Python.

**New features**

* ``setReciprocalExclusions(True)`` computes the terms that subtract the reciprocal-space
  interactions of excluded pairs and exceptions (Ewald, PME and LJPME) in the reciprocal-space
  force group instead of the direct-space one. They only cancel part of the reciprocal-space sum,
  so both should be evaluated together when they are integrated with different time steps
  (Procacci et al., J. Chem. Phys. 108, 8799, 1998; Morrone et al., J. Chem. Theory Comput. 6,
  1798, 2010). With this option, these terms are also kept when ``setIncludeDirectSpace(False)``
  is used. The default, ``False``, keeps the behavior of OpenMM's ``NonbondedForce``.
* A scaling parameter added with ``includeLJDecapping=True`` (seventh argument of ``addScalingParameter``)
  multiplies the Lennard-Jones decapping term of its slice: each Lennard-Jones pair of the slice is split into a
  capped potential, finite at r = 0 and multiplied by the slice's Lennard-Jones scaling parameter, and
  the rest of the usual potential, which is zero beyond sigma (Correa et al., J. Chem. Theory Comput.
  18, 5876, 2022). The energy stays linear in every parameter, so the derivatives are the capped and
  decapping energies of the slice. The polynomial core is defined by the orders of the derivatives that
  vanish at r = 0 and that match the usual potential at r = sigma, both 3 by default
  (``setCappingOrders``). Available on all platforms, with every nonbonded method and with the inner
  force group. The sigma of the Lennard-Jones pairs of decapped slices cannot exceed the cutoff distance,
  nor the switching distance or the inner switching distance when these are used.
* A scaling parameter added with ``includeCoulombDecapping=True`` (sixth argument of
  ``addScalingParameter``) multiplies the Coulomb decapping term of its slice: the 1/r term of the
  direct-space potential of every pair of the slice is replaced below sigma by the polynomial with the
  same orders that matches it at sigma, and the difference is the Coulomb decapping term. One parameter
  can multiply both decapping terms, or each can have its own. The sigma of every pair of charged
  particles of a Coulomb-decapped slice, those with zero epsilon included, cannot exceed the cutoff
  distance. Not available with an inner force group.
* Inner force group: a force-switched short-range copy of the direct-space interactions (particle
  pairs and 1-4 exceptions) can be computed in its own force group for multiple time-step
  integration, with an energy consistent with its force (``setInnerForceGroup``,
  ``setInnerCutoffDistance``, ``setInnerSwitchingDistance``).

**Bug fixes**

* GPU platforms: a ``SlicedNonbondedForce`` without energy parameter derivatives no longer fails to
  compile with PME or LJPME when another force in the same context requests derivatives.
* GPU platforms: after ``updateParametersInContext``, exceptions keep the scaling parameters of
  their own slice instead of those of slice 0.
* OpenCL on non-NVIDIA devices: LJPME with no charges and a requested energy parameter derivative
  no longer fails with ``ComputeArray has not been initialized``.
* OpenCL on CPU devices: the direct-space kernel could read a slice scaling parameter out of bounds,
  using the stale subset of a padding atom in an excluded pair, and crash. The slice index of
  excluded pairs is now fixed to a valid value. Results were not affected.
* OpenCL: the PME/LJPME FFT passed VkFFT a pointer to the command queue handle of a temporary
  object, which could be overwritten before use and then corrupt memory (undefined behavior that
  depended on the compiler and on the surrounding code).

**Performance**

* GPU platforms: PME force interpolation loads the slice scale factors once per atom rather than
  once per grid point, and atoms are sorted every fourth step below 15,000 atoms and every second
  step above, as in OpenMM. A step with 2 subsets in a 1,503-atom water box went from 110 to 64 µs
  (``NonbondedForce``: 58 µs).

**Documentation**

* Benchmark tables on the performance page remeasured with the device-side reduction of energy
  parameter derivatives proposed in `openmm/openmm#5520 <https://github.com/openmm/openmm/pull/5520>`_.
* Documentation reorganized into an overview, an installation page, a user guide, a theory page and a
  performance page; new pages are written in Markdown (MyST).

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
