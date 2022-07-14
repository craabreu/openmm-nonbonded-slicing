%module pmeslicing

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "SlicedPmeForce.h"
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
%pythonappend PmeSlicing::SlicedPmeForce::getExceptionParameters(
        int index, int& particle1, int& particle2, double& chargeProd) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getParticleParameterOffset(
    int index, std::string& parameter, int& particleIndex, double& chargeScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge)
%}

%pythonappend PmeSlicing::SlicedPmeForce::getExceptionParameterOffset(
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


namespace PmeSlicing {

class SlicedPmeForce : public OpenMM::Force {
public:
    SlicedPmeForce(int numSubsets=1);
    SlicedPmeForce(const OpenMM::NonbondedForce&, int numSubsets=1);
    int getNumSubsets() const;
    int getNumParticles() const;
    int getNumExceptions() const;
    int getNumGlobalParameters() const;
    int getNumParticleParameterOffsets() const;
    int getNumExceptionParameterOffsets() const;
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);

    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    void setPMEParameters(double alpha, int nx, int ny, int nz);

    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    int addParticle(double charge, int subset=0);
    int getParticleSubset(int index);
    void setParticleSubset(int index, int subset);
    double getParticleCharge(int index) const;
    void setParticleCharge(int index, double charge);
    int addException(int particle1, int particle2, double chargeProd, bool replace=false);

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
    int getReciprocalSpaceForceGroup() const;
    void setReciprocalSpaceForceGroup(int group);
    bool getIncludeDirectSpace() const;
    void setIncludeDirectSpace(bool include);
    void updateParametersInContext(Context& context);
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);
    int getSliceForceGroup(int subset1, int subset2) const;
    void setSliceForceGroup(int subset1, int subset2, int group);

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

}
