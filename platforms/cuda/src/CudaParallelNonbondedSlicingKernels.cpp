/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "CudaParallelNonbondedSlicingKernels.h"
#include "CudaNonbondedSlicingKernelSources.h"
#include "openmm/common/ContextSelector.h"

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

class CudaParallelCalcSlicedNonbondedForceKernel::Task : public CudaContext::WorkTask {
public:
    Task(ContextImpl& context, CudaCalcSlicedNonbondedForceKernel& kernel, bool includeForce,
            bool includeEnergy, bool includeDirect, bool includeReciprocal, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), includeDirect(includeDirect), includeReciprocal(includeReciprocal), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy, includeDirect, includeReciprocal);
    }
private:
    ContextImpl& context;
    CudaCalcSlicedNonbondedForceKernel& kernel;
    bool includeForce, includeEnergy, includeDirect, includeReciprocal;
    double& energy;
};

CudaParallelCalcSlicedNonbondedForceKernel::CudaParallelCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data, const System& system) :
        CalcSlicedNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CudaCalcSlicedNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}

void CudaParallelCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double CudaParallelCalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        CudaContext& cu = *data.contexts[i];
        ComputeContext::WorkThread& thread = cu.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, includeDirect, includeReciprocal, data.contextEnergy[i]));
    }
    return 0.0;
}

void CudaParallelCalcSlicedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

void CudaParallelCalcSlicedNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const CudaCalcSlicedNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
}

void CudaParallelCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const CudaCalcSlicedNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
}
