/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "OpenCLParallelNonbondedSlicingKernels.h"

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

class OpenCLParallelCalcSlicedNonbondedForceKernel::Task : public OpenCLContext::WorkTask {
public:
    Task(ContextImpl& context, OpenCLCalcSlicedNonbondedForceKernel& kernel, bool includeForce,
            bool includeEnergy, bool includeDirect, bool includeReciprocal, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), includeDirect(includeDirect), includeReciprocal(includeReciprocal), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy, includeDirect, includeReciprocal);
    }
private:
    ContextImpl& context;
    OpenCLCalcSlicedNonbondedForceKernel& kernel;
    bool includeForce, includeEnergy, includeDirect, includeReciprocal;
    double& energy;
};

OpenCLParallelCalcSlicedNonbondedForceKernel::OpenCLParallelCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, OpenCLPlatform::PlatformData& data, const System& system) :
        CalcSlicedNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new OpenCLCalcSlicedNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}

void OpenCLParallelCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double OpenCLParallelCalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        OpenCLContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, includeDirect, includeReciprocal, data.contextEnergy[i]));
    }
    return 0.0;
}

void OpenCLParallelCalcSlicedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

void OpenCLParallelCalcSlicedNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const OpenCLCalcSlicedNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
}

void OpenCLParallelCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const OpenCLCalcSlicedNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
}
