/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "SlicedNonbondedForce.h"
#include "SlicedNonbondedForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_NONBONDED_SLICING void registerNonbondedSlicingSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerNonbondedSlicingSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerNonbondedSlicingSerializationProxies();
#endif

using namespace NonbondedSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_NONBONDED_SLICING void registerNonbondedSlicingSerializationProxies() {
    SerializationProxy::registerProxy(typeid(SlicedNonbondedForce), new SlicedNonbondedForceProxy());
}
