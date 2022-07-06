%module nonbondedslicing

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "SlicedNonbondedForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
from openmm import unit
%}

/*
 * Add units to function outputs.
*/
%pythonappend NonbondedSlicing::SlicedNonbondedForce::getExceptionParameters(
        int index, int& particle1, int& particle2, double& chargeProd) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
%}

%pythonappend NonbondedSlicing::SlicedNonbondedForce::getParticleParameterOffset(
    int index, std::string& parameter, int& particleIndex, double& chargeScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge)
%}

%pythonappend NonbondedSlicing::SlicedNonbondedForce::getExceptionParameterOffset(
    int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
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


namespace NonbondedSlicing {

class SlicedNonbondedForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2,
        Ewald = 3,
        PME = 4,
        LJPME = 5
    };
    SlicedNonbondedForce(int numSubsets=1);
    SlicedNonbondedForce(const OpenMM::NonbondedForce&, int numSubsets=1);
    int getNumSubsets() const;
    int getNumParticles() const;
    int getNumExceptions() const;
    int getNumGlobalParameters() const;
    int getNumParticleParameterOffsets() const;
    int getNumExceptionParameterOffsets() const;
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    bool getUseSwitchingFunction() const;
    void setUseSwitchingFunction(bool use);
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);
    double getReactionFieldDielectric() const;
    void setReactionFieldDielectric(double dielectric);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);

    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    void setPMEParameters(double alpha, int nx, int ny, int nz);
    void setLJPMEParameters(double alpha, int nx, int ny, int nz);

    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    int addParticle(double charge, int subset=0);
    int getParticleSubset(int index);
    void setParticleSubset(int index, int subset);
    double getParticleCharge(int index) const;
    void setParticleCharge(int index, double charge);
    int addException(int particle1, int particle2, double chargeProd, bool replace = false);

    %apply int& OUTPUT {int& particle1};
    %apply int& OUTPUT {int& particle2};
    %apply double& OUTPUT {double& chargeProd};
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd) const;
    %clear int& particle1;
    %clear int& particle2;
    %clear double& chargeProd;

    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd);
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale);
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    int addParticleParameterOffset(const std::string& parameter, int particleIndex, double chargeScale);

    %apply std::string& OUTPUT {std::string& parameter};
    %apply int& OUTPUT {int& particleIndex};
    %apply double& OUTPUT {double& chargeScale};
    void getParticleParameterOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale) const;
    %clear std::string& parameter;
    %clear int& particleIndex;
    %clear double& chargeScale;

    void setParticleParameterOffset(int index, const std::string& parameter, int particleIndex, double chargeScale);
    int addExceptionParameterOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale);

    %apply std::string& OUTPUT {std::string& parameter};
    %apply int& OUTPUT {int& exceptionIndex};
    %apply double& OUTPUT {double& chargeProdScale};
    void getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale) const;
    %clear std::string& parameter;
    %clear int& exceptionIndex;
    %clear double& chargeProdScale;

    void setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale);
    bool getUseDispersionCorrection() const;
    void setUseDispersionCorrection(bool useCorrection);
    int getReciprocalSpaceForceGroup() const;
    void setReciprocalSpaceForceGroup(int group);
    bool getIncludeDirectSpace() const;
    void setIncludeDirectSpace(bool include);
    void updateParametersInContext(Context& context);
    bool usesPeriodicBoundaryConditions() const;
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);
    int getSliceForceGroup(int subset1, int subset2) const;
    void setSliceForceGroup(int subset1, int subset2, int group);

    /*
     * Add methods for casting a Force to a SlicedNonbondedForce.
    */
    %extend {
        static NonbondedSlicing::SlicedNonbondedForce& cast(OpenMM::Force& force) {
            return dynamic_cast<NonbondedSlicing::SlicedNonbondedForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<NonbondedSlicing::SlicedNonbondedForce*>(&force) != NULL);
        }
    }
};

}
