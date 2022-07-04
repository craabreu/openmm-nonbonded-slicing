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
%pythonappend NonbondedSlicing::SlicedNonbondedForce::getParticleParameters(int index, double& charge,
                                                            double& sigma, double& epsilon) const %{
    val[1] = unit.Quantity(val[1], unit.elementary_charge)
    val[2] = unit.Quantity(val[2], unit.nanometer)
    val[3] = unit.Quantity(val[3], unit.kilojoule_per_mole)
%}

%pythonappend NonbondedSlicing::SlicedNonbondedForce::getExceptionParameters(int index, int& particle1,
                        int& particle2, double& chargeProd, double& sigma, double& epsilon) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
    val[4] = unit.Quantity(val[4], unit.nanometer)
    val[5] = unit.Quantity(val[5], unit.kilojoule_per_mole)
%}

%pythonappend NonbondedSlicing::SlicedNonbondedForce::getParticleParameterOffset(int index, std::string& parameter,
         int& particleIndex, double& chargeScale, double& sigmaScale, double& epsilonScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge)
    val[4] = unit.Quantity(val[4], unit.nanometer)
    val[5] = unit.Quantity(val[5], unit.kilojoule_per_mole)
%}

%pythonappend NonbondedSlicing::SlicedNonbondedForce::getExceptionParameterOffset(int index, std::string& parameter,
    int& exceptionIndex, double& chargeProdScale, double& sigmaScale, double& epsilonScale) const %{
    val[3] = unit.Quantity(val[3], unit.elementary_charge**2)
    val[4] = unit.Quantity(val[4], unit.nanometer)
    val[5] = unit.Quantity(val[5], unit.kilojoule_per_mole)
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

    int addParticle(double charge, double sigma, double epsilon, int subset=0);
    int getParticleSubset(int index);
    void setParticleSubset(int index, int subset);

    %apply double& OUTPUT {double& charge};
    %apply double& OUTPUT {double& sigma};
    %apply double& OUTPUT {double& epsilon};
    void getParticleParameters(int index, double& charge, double& sigma, double& epsilon) const;
    %clear double& charge;
    %clear double& sigma;
    %clear double& epsilon;

    void setParticleParameters(int index, double charge, double sigma, double epsilon);
    int addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace = false);

    %apply int& OUTPUT {int& particle1};
    %apply int& OUTPUT {int& particle2};
    %apply double& OUTPUT {double& chargeProd};
    %apply double& OUTPUT {double& sigma};
    %apply double& OUTPUT {double& epsilon};
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const;
    %clear int& particle1;
    %clear int& particle2;
    %clear double& chargeProd;
    %clear double& sigma;
    %clear double& epsilon;

    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon);
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale);
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    int addParticleParameterOffset(const std::string& parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale);

    %apply std::string& OUTPUT {std::string& parameter};
    %apply int& OUTPUT {int& particleIndex};
    %apply double& OUTPUT {double& chargeScale};
    %apply double& OUTPUT {double& sigmaScale};
    %apply double& OUTPUT {double& epsilonScale};
    void getParticleParameterOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale, double& sigmaScale, double& epsilonScale) const;
    %clear std::string& parameter;
    %clear int& particleIndex;
    %clear double& chargeScale;
    %clear double& sigmaScale;
    %clear double& epsilonScale;

    void setParticleParameterOffset(int index, const std::string& parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale);
    int addExceptionParameterOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale);

    %apply std::string& OUTPUT {std::string& parameter};
    %apply int& OUTPUT {int& exceptionIndex};
    %apply double& OUTPUT {double& chargeProdScale};
    %apply double& OUTPUT {double& sigmaScale};
    %apply double& OUTPUT {double& epsilonScale};
    void getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale, double& sigmaScale, double& epsilonScale) const;
    %clear std::string& parameter;
    %clear int& exceptionIndex;
    %clear double& chargeProdScale;
    %clear double& sigmaScale;
    %clear double& epsilonScale;

    void setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale);
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
