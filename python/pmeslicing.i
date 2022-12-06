%module pmeslicing

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "SlicedPmeForce.h"
#include "SlicedNonbondedForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"

#define SWIG_PYTHON_CAST_MODE
%}

%pythoncode %{
from openmm import unit
%}

/*
 * Add units to function outputs.
*/

%pythonappend PmeSlicing::SlicedPmeForce::getCutoffDistance() const %{
    val = unit.Quantity(val, unit.nanometers)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getParticleCharge(int index) const %{
    val = unit.Quantity(val, unit.elementary_charge)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getPMEParameters(
        double& alpha, int& nx, int& ny, int& nz) const %{
    val[0] = unit.Quantity(val[0], 1/unit.nanometers)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getExceptionParameters(
        int index, int& particle1, int& particle2, double& chargeProd) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getParticleChargeOffset(
    int index, std::string& parameter, int& particleIndex, double& chargeScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getExceptionChargeOffset(
    int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
%}

%pythonappend PmeSlicing::SlicedNonbondedForce::getPMEParametersInContext(
        const openMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const %{
    val[0] = unit.Quantity(val[0], 1/unit.nanometers)
%}

%pythonappend PmeSlicing::SlicedNonbondedForce::getLJPMEParametersInContext(
        const openMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const %{
    val[0] = unit.Quantity(val[0], 1/unit.nanometers)
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%apply double& OUTPUT {double& alpha};
%apply int& OUTPUT {int& nx};
%apply int& OUTPUT {int& ny};
%apply int& OUTPUT {int& nz};
%apply int& OUTPUT {int& particle1};
%apply int& OUTPUT {int& particle2};
%apply double& OUTPUT {double& chargeProd};
%apply const std::string& OUTPUT {const std::string& parameter};
%apply int& OUTPUT {int& subset1};
%apply int& OUTPUT {int& subset2};
%apply int& OUTPUT {int& particleIndex};
%apply double& OUTPUT {double& chargeScale};
%apply int& OUTPUT {int& exceptionIndex};
%apply double& OUTPUT {double& chargeProdScale};

namespace PmeSlicing {

/**
 * This class implements a Coulomb force to represent electrostatic interactions between particles
 * under periodic boundary conditions. The computation is done using the smooth Particle Mesh Ewald
 * (PME) method :cite:`Essmann_1995`.
 *
 * The total Coulomb potential can be divided into slices depending on which pairs of particles are
 * involved. After distributing all particles among disjoint subsets, each slice is distinguished
 * by two indices I and J. :math:`U_{I,J}` is the sum of the Coulomb interactions of all particles
 * in subset I with all particles in subset J.
 *
 * To use this class, create a SlicedPmeForce object, then call :func:`addParticle` once for each
 * particle in the System to define its electric charge and its subset. The number of particles
 * for which you define these parameters must be exactly equal to the number of particles in the
 * System, or else an exception will be thrown when you try to create an :OpenMM:`Context`. After
 * a particle has been added, you can modify its electric charge by calling :func:`setParticleCharge`
 * or its subset by calling :func:`setParticleSubset`. This will have no effect on Contexts that
 * already exist unless you call :func:`updateParametersInContext`.
 *
 * :class:`SlicedPmeForce` also lets you specify "exceptions", particular pairs of particles whose
 * interactions should be computed based on a product of charges other than those defined for the
 * individual particles. This can be used to completely exclude certain interactions from the
 * force calculation.
 *
 * Many molecular force fields omit Coulomb interactions between particles separated by one
 * or two bonds, while using modified parameters for those separated by three bonds (known as
 * "1-4 interactions"). This class provides a convenience method for this case called
 * :func:`createExceptionsFromBonds`.  You pass to it a list of bonds and the scale factor to use
 * for 1-4 interactions.  It identifies all pairs of particles which are separated by 1, 2, or
 * 3 bonds, then automatically creates exceptions for them.
 *
 * In some applications, it is useful to be able to inexpensively change the charges of small
 * groups of particles. Usually this is done to interpolate between two sets of parameters. For
 * example, a titratable group might have two states it can exist in, each described by a
 * different set of parameters for the atoms that make up the group. You might then want to
 * smoothly interpolate between the two states. This is done by first calling addGlobalParameter`
 * to define a Context parameter, then :func:`addParticleChargeOffset` to create a "charge offset"
 * that depends on the :OpenMM:`Context` parameter. Each offset defines the following:
 *
 * * A Context parameter used to interpolate between the states.
 * * A single particle whose parameters are influenced by the :OpenMM:`Context` parameter.
 * * A scale factor (chargeScale) that specifies how the :OpenMM:`Context` parameter affects the particle.
 *
 * The "effective" charge of a particle (that used to compute forces) is given by
 *
 * .. code-block:: python
 *
 *    charge = baseCharge + param*chargeScale
 *
 * where the "base" values are the ones specified by :func:`addParticle` and "param" is the current
 * value of the :OpenMM:`Context` parameter. A single :OpenMM:`Context` parameter can apply offsets
 * to multiple particles, and multiple parameters can be used to apply offsets to the same
 * particle. Parameters can also be used to modify exceptions in exactly the same way by calling
 * :func:`addExceptionChargeOffset`.
 */
class SlicedPmeForce : public OpenMM::Force {
public:
    /**
     * Create a SlicedPmeForce.
     *
     * Parameters
     * ----------
     *     numSubsets : int
     *         the number of particle subsets
     */
    SlicedPmeForce(int numSubsets);
    /**
     * Create a SlicedPmeForce whose properties are imported from an existing NonbondedForce.
     *
     * Parameters
     * ----------
     *     nonbondedForce : :Openmm:`NonbondedForce`
     *         the NonbondedForce whose properties will be imported
     *     numSubsets : int
     *         the number of particle subsets
     */
    SlicedPmeForce(const OpenMM::NonbondedForce&, int numSubsets);
    /**
     * Get the specified number of particle subsets.
     */
    int getNumSubsets() const;
    /**
     * Get the number of particles for which force field parameters have been defined.
     */
    int getNumParticles() const;
    /**
     * Get the number of special interactions that should be calculated differently from other
     * interactions.
     */
    int getNumExceptions() const;
    /**
     * Get the number of global parameters that have been added.
     */
    int getNumGlobalParameters() const;
    /**
     * Get the number of particles charge offsets that have been added.
     */
    int getNumParticleChargeOffsets() const;
    /**
     * Get the number of exception charge offsets that have been added.
     */
    int getNumExceptionChargeOffsets() const;
    /**
     * Get the cutoff distance (in :math:`nm`) being used for nonbonded interactions.
     *
     * Returns
     * -------
     *     cutoff : double
     *         the cutoff distance, measured in :math:`nm`
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in :math:`nm`) being used for nonbonded interactions.
     *
     * Parameters
     * ----------
     *      distance : double
     *          the cutoff distance, measured in :math:`nm`
     */
    void setCutoffDistance(double distance);
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in
     * the forces which is acceptable.  This value is used to select the reciprocal space cutoff
     * and separation parameter so that the average error level will be less than the tolerance.
     * There is not a rigorous guarantee that all forces on all atoms will be less than the
     * tolerance, however.
     *
     * For PME calculations, if :func:`setPMEParameters` is used to set alpha to something other than 0,
     * this value is ignored.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in
     * the forces which is acceptable.  This value is used to select the reciprocal space cutoff
     * and separation parameter so that the average error level will be less than the tolerance.
     * There is not a rigorous guarantee that all forces on all atoms will be less than the
     * tolerance, however.
     *
     * For PME calculations, if :func:`setPMEParameters` is used to set alpha to something other
     * than 0, this value is ignored.
     *
     * Parameters
     * ----------
     *     tol : double
     *         the fractional error tolerance for Ewald summation
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the parameters to use for PME calculations. If alpha is 0 (the default),
     * these parameters are ignored and instead their values are chosen based on the Ewald error
     * tolerance.
     *
     * Returns
     * -------
     *     alpha : double
     *         the separation parameter, measured in :math:`nm^{-1}`
     *     nx : int
     *         the number of grid points along the X axis
     *     ny : int
     *         the number of grid points along the Y axis
     *     nz : int
     *         the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these
     * parameters are ignored and instead their values are chosen based on the Ewald error
     * tolerance.
     *
     * Parameters
     * ----------
     *     alpha : double
     *         the separation parameter, measured in :math:`nm^{-1}`
     *     nx : int
     *         the number of grid points along the X axis
     *     ny : int
     *         the number of grid points along the Y axis
     *     nz : int
     *         the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have
     * restrictions on the allowed grid sizes, the values that are actually used may be slightly
     * different from those specified with :func:`setPMEParameters`, or the standard values calculated
     * based on the Ewald error tolerance. See the manual for details.
     *
     * Parameters
     * ----------
     *     context : :OpenMM:`Context`
     *         the Context for which to get the parameters
     *
     * Returns
     * -------
     *     alpha : double
     *         the separation parameter, measured in :math:`nm^{-1}`
     *     nx : int
     *         the number of grid points along the X axis
     *     ny : int
     *         the number of grid points along the Y axis
     *     nz : int
     *         the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Add the charges and (optionally) the subset for a particle.  This should be called once
     * for each particle in the System.  When it is called for the i'th time, it specifies the
     * charge for the i'th particle.
     *
     * Parameters
     * ----------
     *     charge : double
     *         the charge of the particle, measured in units of the proton charge
     *     subset : int, optional
     *         the subset to which this particle belongs (default=0)
     *
     * Returns
     * -------
     *     index : int
     *         the index of the particle that was added
     */
    int addParticle(double charge, int subset=0);
    /**
     * Get the subset to which a particle belongs.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the particle for which to get the subset
     */
    int getParticleSubset(int index) const;
    /**
     * Set the subset for a particle.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the particle for which to set the subset
     *     subset : int
     *         the subset to which this particle belongs
     */
    void setParticleSubset(int index, int subset);
    /**
     * Get the charge of a particle.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the particle for which to get parameters
     *
     * Returns
     * -------
     *     charge : double
     *         the charge of the particle, measured in units of the proton charge
     */
    double getParticleCharge(int index) const;
    /**
     * Set the charge for a particle.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the particle for which to set parameters
     *     charge : double
     *         the charge of the particle, measured in units of the proton charge
     */
    void setParticleCharge(int index, double charge);
    /**
     * Add an interaction to the list of exceptions that should be calculated differently from
     * other interactions. If chargeProd is equal to 0, this will cause the interaction to be
     * completely omitted from force and energy calculations.
     *
     * Cutoffs are never applied to exceptions. That is because they are primarily used for 1-4
     * interactions, which are really a type of bonded interaction and are parametrized together
     * with the other bonded interactions.
     *
     * In many cases, you can use :func:`createExceptionsFromBonds` rather than adding each exception
     * explicitly.
     *
     * Parameters
     * ----------
     *     particle1 : int
     *         the index of the first particle involved in the interaction
     *     particle2 : int
     *         the index of the second particle involved in the interaction
     *     chargeProd : double
     *         the scaled product of the atomic charges (i.e. the strength of the Coulomb
     *         interaction), measured in units of the proton charge squared
     *     replace : bool, optional
     *         determines the behavior if there is already an exception for the same two
     *         particles. If `True`, the existing one is replaced. If `False`, an exception
     *         is thrown.
     *
     * Returns
     * -------
     *     index : int
     *         the index of the exception that was added
     */
    int addException(int particle1, int particle2, double chargeProd, bool replace = false);
    /**
     * Get the particle indices and charge product for an interaction that should be calculated
     * differently from others.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the interaction for which to get parameters
     *
     * Returns
     * -------
     *     particle1 : int
     *         the index of the first particle involved in the interaction
     *     particle2 : int
     *         the index of the second particle involved in the interaction
     *     chargeProd : double
     *         the scaled product of the atomic charges (i.e. the strength of the
     *         Coulomb interaction), measured in units of the proton charge squared
     */
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd) const;
    /**
     * Set the particle indices and charge product for an interaction that should be calculated
     * differently from others. If chargeProd is equal to 0, this will cause the interaction to be
     * completely omitted from force and energy calculations.
     *
     * Cutoffs are never applied to exceptions. That is because they are primarily used for 1-4
     * interactions, which are really a type of bonded interaction and are parametrized together
     * with the other bonded interactions.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the interaction for which to get parameters
     *     particle1 : int
     *         the index of the first particle involved in the interaction
     *     particle2 : int
     *         the index of the second particle involved in the interaction
     *     chargeProd : double
     *         the scaled product of the atomic charges (i.e. the strength of the Coulomb
     *         interaction), measured in units of the proton charge squared
     */
    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd);
    /**
     * Identify exceptions based on the molecular topology.  Particles which are separated by one
     * or two bonds are set to not interact at all, while pairs of particles separated by three
     * bonds (known as "1-4 interactions") have their Coulomb and Lennard-Jones interactions
     * reduced by a fixed factor.
     *
     * Parameters
     * ----------
     *     bonds : list of (int, int) tuples
     *         the set of bonds based on which to construct exceptions. Each element
     *         specifies the indices of two particles that are bonded to each other.
     *     coulomb14Scale : double
     *         pairs of particles separated by three bonds will have the strength of
     *         their Coulomb interaction multiplied by this factor
     *     lj14Scale : double
     *         pairs of particles separated by three bonds will have the strength of
     *         their Lennard-Jones interaction multiplied by this factor
     */
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale);
    /**
     * Add a new global parameter that charge offsets may depend on.  The default value provided
     * to this method is the initial value of the parameter in newly created Contexts.  You can
     * change the value at any time by calling `setParameter()` on the :OpenMM:`Context`.
     *
     * Parameters
     * ----------
     *     name             the name of the parameter
     *     defaultValue     the default value of the parameter
     *
     * Returns
     * -------
     *     index : int
     *         the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter for which to get the name
     *
     * Returns
     * -------
     *     name : str
     *         the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter for which to set the name
     *     name : str
     *         the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter for which to get the default value
     *
     * Returns
     * -------
     *     defaultValue : double
     *         the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter for which to set the default value
     *     defaultValue: double
     *         the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
 	/**
     * Add a switching parameter to multiply a particular Coulomb slice. Its value will scale the
     * Coulomb interactions between particles of a subset 1 with those of another (or the same)
     * subset 2. The order of subset definition is irrelevant.
     *
     * Parameters
     * ----------
     *     parameter : str
     *         the name of the global parameter.  It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`
     *     subset1 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *     subset2 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *
     * Returns
     * -------
     *     index : int
     *         the index of switching parameter that was added
     */
    int addSwitchingParameter(const std::string& parameter, int subset1, int subset2);
    /**
     * Get the number of switching parameters.
     */
    int getNumSwitchingParameters() const;
  	/**
     * Get the switching parameter applied to a particular nonbonded slice.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the switching parameter to query, as returned by :func:`addSwitchingParameter`
     *
     * Returns
     * -------
     *     parameter : str
     *         the name of the global parameter
     *     subset1 : int
     *         the smallest index of the two particle subsets
     *     subset2 : int
     *         the largest index of the two particle subsets
     */
    void getSwitchingParameter(int index, std::string& parameter, int& subset1, int& subset2) const;
 	/**
     * Modify an added switching parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the switching parameter to modify, as returned by
     *         :func:`addExceptionChargeOffset`
     *     parameter : str
     *         the name of the global parameter.  It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`
     *     subset1 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *     subset2 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     */
    void setSwitchingParameter(int index, const std::string& parameter, int subset1, int subset2);
    /**
     * Request the derivative of this Force's energy with respect to a switching parameter. This
     * can be used to obtain the sum of particular energy slices. The parameter must have already
     * been added with :func:`addGlobalParameter` and :func:`addSwithingParameter`.
     *
     * Parameters
     * ----------
     *     parameter : str
     *         the name of the parameter
     *
     * Returns
     * -------
     *     index : int
     *         the index of switching parameter derivative that was added
     */
    int addSwitchingParameterDerivative(const std::string& parameter);
    /**
     * Get the number of requested switching parameter derivatives.
     */
    int getNumSwitchingParameterDerivatives() const;
    /**
     * Get the name of the global parameter associated with a requested switching parameter
     * derivative.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter derivative, between 0 and the result of
     *         :func:`getNumSwitchingParameterDerivatives`
     *
     * Returns
     * -------
     *     parameter : str
     *         the parameter name
     */
    const std::string& getSwitchingParameterDerivativeName(int index) const;
    /**
     * Set the name of the global parameter to associate with a requested switching parameter
     * derivative.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter derivative, between 0 and getNumSwitchingParameterDerivatives`
     *     parameter : str
     *         the name of the parameter
     */
    void setSwitchingParameterDerivative(int index, const std::string& parameter);
    /**
     * Add an offset to the charge of a particular particle, based on a global parameter.
     *
     * Parameters
     * ----------
     *     parameter : str
     *         the name of the global parameter. It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`.
     *     particleIndex : int
     *         the index of the particle whose parameters are affected
     *     chargeScale : double
     *         this value multiplied by the parameter value is added to the particle's charge
     *
     * Returns
     * -------
     *     index : int
     *         the index of the offset that was added
     */
    int addParticleChargeOffset(const std::string& parameter, int particleIndex, double chargeScale);
    /**
     * Get the offset added to the per-particle parameters of a particular particle, based on a
     * global parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the offset to query, as returned by :func:`addParticleChargeOffset`
     *
     * Returns
     * -------
     *     parameter : str
     *         the name of the global parameter
     *     particleIndex : int
     *         the index of the particle whose parameters are affected
     *     chargeScale : double
     *         this value multiplied by the parameter value is added to the particle's charge
     */
    void getParticleChargeOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale) const;
    /**
     * Set the offset added to the per-particle parameters of a particular particle, based on a
     * global parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the offset to query, as returned by :func:`addParticleChargeOffset`
     *     parameter : str
     *         the name of the global parameter. It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`.
     *     particleIndex : int
     *         the index of the particle whose parameters are affected
     *     chargeScale : double
     *         this value multiplied by the parameter value is added to the particle's charge
     */
    void setParticleChargeOffset(int index, const std::string& parameter, int particleIndex, double chargeScale);
    /**
     * Add an offset to the parameters of a particular exception, based on a global parameter.
     *
     * Parameters
     * ----------
     *     parameter : str
     *         the name of the global parameter.  It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`.
     *     exceptionIndex : int
     *         the index of the exception whose parameters are affected
     *     chargeProdScale : double
     *         this value multiplied by the parameter value is added to the exception's charge product
     *
     * Returns
     * -------
     *     index : int
     *         the index of the offset that was added
     */
    int addExceptionChargeOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale);
    /**
     * Get the offset added to the parameters of a particular exception, based on a global
     * parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the offset to query, as returned by :func:`addExceptionChargeOffset`
     *
     * Returns
     * -------
     *     parameter : str
     *         the name of the global parameter
     *     exceptionIndex : int
     *         the index of the exception whose parameters are affected
     *     chargeProdScale : double
     *         this value multiplied by the parameter value is added to the exception's charge product
     */
    void getExceptionChargeOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale) const;
    /**
     * Set the offset added to the parameters of a particular exception, based on a global
     * parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the offset to modify, as returned by :func:`addExceptionChargeOffset`
     *     parameter : str
     *         the name of the global parameter.  It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`.
     *     exceptionIndex : int
     *         the index of the exception whose parameters are affected
     *     chargeProdScale : double
     *         this value multiplied by the parameter value is added to the exception's charge product
     */
    void setExceptionChargeOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale);
    /**
     * Get the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: getForceGroup`
     * specifies the group for direct space, and :func:`getReciprocalSpaceForceGroup` specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     */
    int getReciprocalSpaceForceGroup() const;
    /**
     * Set the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: setForceGroup`
     * specifies the group for direct space, and :func:`setReciprocalSpaceForceGroup` specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     *
     * Parameters
     * ----------
     *     group : int
     *         the group index.  Legal values are between 0 and 31 (inclusive), or -1 to use the same force group
     *         that is specified for direct space.
     */
    void setReciprocalSpaceForceGroup(int group);
    /**
     * Get whether to include direct space interactions when calculating forces and energies.  This is useful if you want
     * to completely replace the direct space calculation, typically with a CustomNonbondedForce that computes it in a
     * nonstandard way, while still using this object for the reciprocal space calculation.
     */
    bool getIncludeDirectSpace() const;
    /**
     * Set whether to include direct space interactions when calculating forces and energies.  This is useful if you want
     * to completely replace the direct space calculation, typically with a CustomNonbondedForce that computes it in a
     * nonstandard way, while still using this object for the reciprocal space calculation.
     */
    void setIncludeDirectSpace(bool include);
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call :func:`setParticleCharge` and :func:`setExceptionParameters` to modify this object's parameters, then call
     * :func:`updateParametersInContext` to copy them over to the :OpenMM:`Context`.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the :OpenMM:`Context`.  Furthermore, only the chargeProd, sigma, and epsilon values of an exception
     * can be changed; the pair of particles involved in the exception cannot change.  Finally, this method cannot be used
     * to add new particles or exceptions, only to change the parameters of existing ones.
     *
     * Parameters
     * ----------
     *     context : :OpenMM:`Context`
     *         the Context whose parameters should be updated
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Get whether periodic boundary conditions should be applied to exceptions.  Usually this is not
     * appropriate, because exceptions are normally used to represent bonded interactions (1-2, 1-3, and
     * 1-4 pairs), but there are situations when it does make sense.  For pmeslicing, you may want to simulate
     * an infinite chain where one end of a molecule is bonded to the opposite end of the next periodic
     * copy.
     *
     * Regardless of this value, periodic boundary conditions are only applied to exceptions if they also
     * are applied to other interactions.  If the nonbonded method is NoCutoff or CutoffNonPeriodic, this
     * value is ignored.  Also note that cutoffs are never applied to exceptions, again because they are
     * normally used to represent bonded interactions.
     */
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    /**
     * Set whether periodic boundary conditions should be applied to exceptions. Usually this is
     * not appropriate, because exceptions are normally used to represent bonded interactions
     * (1-2, 1-3, and 1-4 pairs), but there are situations when it does make sense.  For example,
     * you may want to simulate an infinite chain where one end of a molecule is bonded to the
     * opposite end of the next periodic copy.
     *
     * Regardless of this value, periodic boundary conditions are only applied to exceptions if
     * they also get applied to other interactions.  If the nonbonded method is NoCutoff or
     * CutoffNonPeriodic, this value is ignored.  Also note that cutoffs are never applied to
     * exceptions, again because they are normally used to represent bonded interactions.
     *
     * Parameters
     * ----------
     *     periodic : bool
     *         whether to apply periodic boundary conditions to exceptions
     */
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);
 	/**
     * Get whether to use CUDA Toolkit's cuFFT library when executing in the CUDA platform.
     * The default value is `False`.
     */
    bool getUseCudaFFT() const;
 	/**
     * Set whether whether to use CUDA Toolkit's cuFFT library when executing in the CUDA platform.
     * This choice has no effect when using other platforms or when the CUDA Toolkit is version 7.0
     * or older.
     *
     * Parameters
     * ----------
     *     use : bool
     *         whether to use the cuFFT library
     */
    void setUseCuFFT(bool use);

    /*
     * Add methods for casting a Force to a SlicedPmeForce.
    */

    %extend {
        static PmeSlicing::SlicedPmeForce& cast(OpenMM::Force& force) {
            return dynamic_cast<PmeSlicing::SlicedPmeForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<PmeSlicing::SlicedPmeForce*>(&force) != NULL);
        }
    }
};

%clear double& alpha;
%clear int& nx;
%clear int& ny;
%clear int& nz;
%clear int& particle1;
%clear int& particle2;
%clear double& chargeProd;
%clear std::string& parameter;
%clear int& subset1;
%clear int& subset2;
%clear int& particleIndex;
%clear double& chargeScale;
%clear int& exceptionIndex;
%clear double& chargeProdScale;

%apply double& OUTPUT {double& alpha};
%apply int& OUTPUT {int& nx};
%apply int& OUTPUT {int& ny};
%apply int& OUTPUT {int& nz};
%apply const std::string& OUTPUT {const std::string& parameter};
%apply int& OUTPUT {int& subset1};
%apply int& OUTPUT {int& subset2};
%apply bool& OUTPUT {bool& includeLJ};
%apply bool& OUTPUT {bool& includeCoulomb};

/**
 * This class implements nonbonded interactions between particles, including a Coulomb force to represent
 * electrostatics and a Lennard-Jones force to represent van der Waals interactions.  It optionally supports
 * periodic boundary conditions and cutoffs for long range interactions.  Lennard-Jones interactions are
 * calculated with the Lorentz-Berthelot combining rule: it uses the arithmetic mean of the sigmas and the
 * geometric mean of the epsilons for the two interacting particles.
 *
 * To use this class, create a NonbondedForce object, then call :func:`addParticle` once for each particle in the
 * System to define its parameters.  The number of particles for which you define nonbonded parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its force field parameters
 * by calling :func:`setParticleParameters`.  This will have no effect on Contexts that already exist unless you
 * call :func:`updateParametersInContext`.
 *
 * NonbondedForce also lets you specify "exceptions", particular pairs of particles whose interactions should be
 * computed based on different parameters than those defined for the individual particles.  This can be used to
 * completely exclude certain interactions from the force calculation, or to alter how they interact with each other.
 *
 * Many molecular force fields omit Coulomb and Lennard-Jones interactions between particles separated by one
 * or two bonds, while using modified parameters for those separated by three bonds (known as "1-4 interactions").
 * This class provides a convenience method for this case called :func:`createExceptionsFromBonds`.  You pass to it
 * a list of bonds and the scale factors to use for 1-4 interactions.  It identifies all pairs of particles which
 * are separated by 1, 2, or 3 bonds, then automatically creates exceptions for them.
 *
 * When using a cutoff, by default Lennard-Jones interactions are sharply truncated at the cutoff distance.
 * Optionally you can instead use a switching function to make the interaction smoothly go to zero over a finite
 * distance range.  To enable this, call :func:`setUseSwitchingFunction`.  You must also call :func:`setSwitchingDistance`
 * to specify the distance at which the interaction should begin to decrease.  The switching distance must be
 * less than the cutoff distance.
 *
 * Another optional feature of this class (enabled by default) is to add a contribution to the energy which approximates
 * the effect of all Lennard-Jones interactions beyond the cutoff in a periodic system.  When running a simulation
 * at constant pressure, this can improve the quality of the result.  Call :func:`setUseDispersionCorrection` to set whether
 * this should be used.
 *
 * In some applications, it is useful to be able to inexpensively change the parameters of small groups of particles.
 * Usually this is done to interpolate between two sets of parameters.  For example, a titratable group might have
 * two states it can exist in, each described by a different set of parameters for the atoms that make up the
 * group.  You might then want to smoothly interpolate between the two states.  This is done by first calling
 * :func:`addGlobalParameter` to define a Context parameter, then :func:`addParticleParameterOffset` to create a "parameter offset"
 * that depends on the Context parameter.  Each offset defines the following:
 *
 * * A Context parameter used to interpolate between the states.
 * * A single particle whose parameters are influenced by the Context parameter.
 * * Three scale factors (chargeScale, sigmaScale, and epsilonScale) that specify how the Context parameter affects the particle.
 *
 * The "effective" parameters for a particle (those used to compute forces) are given by
 *
 * .. code-block:: python
 *
 *    charge = baseCharge + param*chargeScale
 *    sigma = baseSigma + param*sigmaScale
 *    epsilon = baseEpsilon + param*epsilonScale
 *
 * where the "base" values are the ones specified by :func:`addParticle` and "oaram" is the current value
 * of the Context parameter.  A single Context parameter can apply offsets to multiple particles,
 * and multiple parameters can be used to apply offsets to the same particle.  Parameters can also be used
 * to modify exceptions in exactly the same way by calling :func:`addExceptionParameterOffset`.
 */
class SlicedNonbondedForce : public OpenMM::NonbondedForce {
public:
    /**
     * Create a SlicedNonbondedForce.
     *
     * Parameters
     * ----------
     *     numSubsets : int
     *         the number of particle subsets
     */
    SlicedNonbondedForce(int numSubsets);
    /**
     * Create a SlicedNonbondedForce having the properties of an existing :OpenMM:`NonbondedForce`.
     *
     * Parameters
     * ----------
     *     force : :OpenMM:`NonbondedForce`
     *         the NonbondedForce object from which to instantiate this SlicedNonbondedForce
     *     numSubsets : int
     *         the number of particle subsets
     */
    SlicedNonbondedForce(const OpenMM::NonbondedForce& force, int numSubsets);
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * Parameters
     * ----------
     *     context : Context
     *         the Context for which to get the parameters
     *
     * Returns
     * -------
     *     alpha : double
     *         the separation parameter, measured in :math:`nm^{-1}`
     *     nx : int
     *         the number of grid points along the X axis
     *     ny : int
     *         the number of grid points along the Y axis
     *     nz : int
     *         the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the PME parameters being used for the dispersion term for LJPME in a particular Context.  Because some
     * platforms have restrictions on the allowed grid sizes, the values that are actually used may be slightly different
     * from those specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * Parameters
     * ----------
     *     context : Context
     *         the Context for which to get the parameters
     *
     * Returns
     * -------
     *     alpha : double
     *         the separation parameter, measured in :math:`nm^{-1}`
     *     nx : int
     *         the number of grid points along the X axis
     *     ny : int
     *         the number of grid points along the Y axis
     *     nz : int
     *         the number of grid points along the Z axis
     */
    void getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() and setExceptionParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, only the chargeProd, sigma, and epsilon values of an exception
     * can be changed; the pair of particles involved in the exception cannot change.  Finally, this method cannot be used
     * to add new particles or exceptions, only to change the parameters of existing ones.
     *
     * Parameters
     * ----------
     *     context : Context
     *         the Context in which to update the parameters
     */
    void updateParametersInContext(Context& context);
    /**
     * Get the specified number of particle subsets.
     */
    int getNumSubsets() const;
    /**
     * Get the number of slices determined by the specified number of particle subsets.
     */
    int getNumSlices() const;
    /**
     * Get the subset to which a particle belongs.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the particle for which to get the subset
     */
    int getParticleSubset(int index) const;
    /**
     * Set the subset of a particle.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the particle for which to set the subset
     *     subset : int
     *         the subset to which this particle belongs
     */
    void setParticleSubset(int index, int subset);
  	/**
     * Add a scaling parameter to multiply a particular Coulomb slice. Its value will scale the
     * Coulomb interactions between particles of a subset 1 with those of another (or the same)
     * subset 2. The order of subset definition is irrelevant.
     *
     * Parameters
     * ----------
     *     parameter : str
     *         the name of the global parameter.  It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`
     *     subset1 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *     subset2 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *     includeLJ : bool
     *         whether this scaling parameter applies to Lennard-Jones interactions
     *     includeCoulomb : bool
     *         whether this scaling parameter applies to Coulomb interactions
     *
     * Returns
     * -------
     *     index : int
     *         the index of scaling parameter that was added
     */
    int addScalingParameter(const std::string& parameter, int subset1, int subset2, bool includeLJ, bool includeCoulomb);
    /**
     * Get the number of scaling parameters.
     */
    int getNumScalingParameters() const;
  	/**
     * Get the scaling parameter applied to a particular nonbonded slice.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the scaling parameter to query, as returned by :func:`addScalingParameter`
     *
     * Returns
     * -------
     *     parameter : str
     *         the name of the global parameter
     *     subset1 : int
     *         the smallest index of the two particle subsets
     *     subset2 : int
     *         the largest index of the two particle subsets
     *     includeLJ : bool
     *         whether this scaling parameter applies to Lennard-Jones interactions
     *     includeCoulomb : bool
     *         whether this scaling parameter applies to Coulomb interactions
     */
    void getScalingParameter(int index, std::string& parameter, int& subset1, int& , bool& includeLJ, bool& includeCoulomb) const;
 	/**
     * Modify an added scaling parameter.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the scaling parameter to modify, as returned by
     *         :func:`addExceptionChargeOffset`
     *     parameter : str
     *         the name of the global parameter.  It must have already been added
     *         with :func:`addGlobalParameter`. Its value can be modified at any time by
     *         calling `setParameter()` on the :OpenMM:`Context`
     *     subset1 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *     subset2 : int
     *         the index of a particle subset.  Legal values are between 0 and the result of
     *         :func:`getNumSubsets`
     *     includeLJ : bool
     *         whether this scaling parameter applies to Lennard-Jones interactions
     *     includeCoulomb : bool
     *         whether this scaling parameter applies to Coulomb interactions
     */
    void setScalingParameter(int index, const std::string& parameter, int subset1, int subset2, bool includeLJ, bool includeCoulomb);
    /**
     * Request the derivative of this Force's energy with respect to a scaling parameter. This
     * can be used to obtain the sum of particular energy slices. The parameter must have already
     * been added with :func:`addGlobalParameter` and :func:`addSwithingParameter`.
     *
     * Parameters
     * ----------
     *     parameter : str
     *         the name of the parameter
     *
     * Returns
     * -------
     *     index : int
     *         the index of scaling parameter derivative that was added
     */
    int addScalingParameterDerivative(const std::string& parameter);
    /**
     * Get the number of requested scaling parameter derivatives.
     */
    int getNumScalingParameterDerivatives() const;
    /**
     * Get the name of the global parameter associated with a requested scaling parameter
     * derivative.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter derivative, between 0 and the result of
     *         :func:`getNumScalingParameterDerivatives`
     */
    const std::string& getScalingParameterDerivativeName(int index) const;
    /**
     * Set the name of the global parameter to associate with a requested scaling parameter
     * derivative.
     *
     * Parameters
     * ----------
     *     index : int
     *         the index of the parameter derivative, between 0 and getNumScalingParameterDerivatives`
     *     parameter : str
     *         the name of the parameter
     */
    void setScalingParameterDerivative(int index, const std::string& parameter);
	/**
     * Get whether to use CUDA Toolkit's cuFFT library when executing in the CUDA platform.
     * The default value is `False`.
     */
    bool getUseCudaFFT() const;
 	/**
     * Set whether whether to use CUDA Toolkit's cuFFT library when executing in the CUDA platform.
     * This choice has no effect when using other platforms or when the CUDA Toolkit is version 7.0
     * or older.
     *
     * Parameters
     * ----------
     *     use : bool
     *         whether to use the cuFFT library
     */
    void setUseCuFFT(bool use);

    /*
     * Add methods for casting a Force to a SlicedNonbondedForce.
    */

    %extend {
        static PmeSlicing::SlicedNonbondedForce& cast(OpenMM::Force& force) {
            return dynamic_cast<PmeSlicing::SlicedNonbondedForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<PmeSlicing::SlicedNonbondedForce*>(&force) != NULL);
        }
    }
};

%clear double& alpha;
%clear int& nx;
%clear int& ny;
%clear int& nz;
%clear std::string& parameter;
%clear int& subset1;
%clear int& subset2;
%clear bool& includeLJ;
%clear bool& includeCoulomb;

}
