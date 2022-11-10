/* -------------------------------------------------------------------------- *
 *                                OpenMMPmeSlicing                                 *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "SlicedPmeForce.h"
#include "SlicedPmeForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_PMESLICING void registerPmeSlicingSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerPmeSlicingSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerPmeSlicingSerializationProxies();
#endif

using namespace PmeSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_PMESLICING void registerPmeSlicingSerializationProxies() {
    SerializationProxy::registerProxy(typeid(SlicedPmeForce), new SlicedPmeForceProxy());
}
