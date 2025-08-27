#---------------------------------------------------
# OpenMM NonbondedSlicing Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(OpenMMNonbondedSlicing VERSION 0.2.2)

# Set CMake policies
CMAKE_POLICY(SET CMP0146 NEW)  # Modern CUDA handling
CMAKE_POLICY(SET CMP0175 NEW)  # Enforce valid add_custom_command() arguments

IF(DEFINED ENV{CONDA_PREFIX})
    SET(OPENMM_DIR $ENV{CONDA_PREFIX} CACHE PATH "Where OpenMM is installed")
    EXECUTE_PROCESS(
        COMMAND conda list ^openmm\\b
        OUTPUT_VARIABLE CONDA_OPENMM_VERSION
    )
    IF (NOT CONDA_OPENMM_VERSION)
        MESSAGE(FATAL_ERROR "OpenMM conda package not found. Please install OpenMM with conda.")
    ENDIF(NOT CONDA_OPENMM_VERSION)
    STRING(REGEX REPLACE "[^\\.]+([0-9]+\\.[0-9]+).*" "\\1" CONDA_OPENMM_VERSION "${CONDA_OPENMM_VERSION}")
    SET(OPENMM_VERSION ${CONDA_OPENMM_VERSION} CACHE STRING "OpenMM version")
ELSE(DEFINED ENV{CONDA_PREFIX})
    MESSAGE(FATAL_ERROR "CONDA_PREFIX is not set. Please set it to the conda environment where OpenMM is installed.")
ENDIF(DEFINED ENV{CONDA_PREFIX})

MESSAGE(STATUS "OpenMM version: ${OPENMM_VERSION}")
STRING(REGEX REPLACE "^([0-9]+)\\.[0-9]+.*" "\\1" OPENMM_VERSION_MAJOR "${OPENMM_VERSION}")
STRING(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" OPENMM_VERSION_MINOR "${OPENMM_VERSION}")
MESSAGE(STATUS "OpenMM major version: ${OPENMM_VERSION_MAJOR}")
MESSAGE(STATUS "OpenMM minor version: ${OPENMM_VERSION_MINOR}")
IF(OPENMM_VERSION_MAJOR LESS 8)
    MESSAGE(FATAL_ERROR "OpenMM version must be at least 8.0")
ENDIF()

ADD_COMPILE_DEFINITIONS(OPENMM_VERSION_MAJOR=${OPENMM_VERSION_MAJOR})
ADD_COMPILE_DEFINITIONS(OPENMM_VERSION_MINOR=${OPENMM_VERSION_MINOR})

INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 11)

# Set flags for linking on mac
IF(APPLE)
    SET(CMAKE_INSTALL_NAME_DIR "@rpath")
    # Check if we're on Apple Silicon (ARM64) or Intel
    EXECUTE_PROCESS(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(ARCH STREQUAL "arm64")
        SET(EXTRA_COMPILE_FLAGS "-stdlib=libc++")
    ELSE()
        SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
    ENDIF()
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX} CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(PLUGIN_LIBRARY_NAME NonbondedSlicing)
SET(SHARED_NONBONDED_SLICING_TARGET ${PLUGIN_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.

ADD_LIBRARY(${SHARED_NONBONDED_SLICING_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_NONBONDED_SLICING_TARGET}
    PROPERTIES COMPILE_FLAGS "-DPLUGIN_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_NONBONDED_SLICING_TARGET} OpenMM)
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_NONBONDED_SLICING_TARGET})

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing

ENABLE_TESTING()
ADD_SUBDIRECTORY(serialization/tests)
INCLUDE(CTest)

# Modern OpenCL detection using OpenCLHeaders
find_package(OpenCLHeaders QUIET)
if(OpenCLHeaders_FOUND)
    MESSAGE("Found OpenCL: TRUE")
    # Set OpenCL variables for compatibility
    SET(OPENCL_FOUND TRUE)
    SET(OPENCL_INCLUDE_DIR ${OpenCLHeaders_INCLUDE_DIRS})
    SET(OPENCL_LIBRARY "OpenCL")
    SET(PLUGIN_BUILD_OPENCL_LIB ON CACHE BOOL "Build implementation for OpenCL")
    SET(PLUGIN_BUILD_OPENCL_SUPPRESS_WARNINGS ON CACHE BOOL "Suppress vkFFT deprecated function warnings in OpenCL build")
ELSE()
    MESSAGE("Found OpenCL: FALSE")
    SET(PLUGIN_BUILD_OPENCL_LIB OFF CACHE BOOL "Build implementation for OpenCL")
ENDIF()

INCLUDE(CheckLanguage)
CHECK_LANGUAGE(CUDA)

IF(CMAKE_CUDA_COMPILER)
    ENABLE_LANGUAGE(CUDA)
    MESSAGE("Found CUDA: TRUE")
    SET(PLUGIN_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
    
    # Find CUDA toolkit directory from compiler path
    GET_FILENAME_COMPONENT(CUDA_COMPILER_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    GET_FILENAME_COMPONENT(CUDA_TOOLKIT_ROOT "${CUDA_COMPILER_DIR}/.." ABSOLUTE)
    
    IF(EXISTS "${CUDA_TOOLKIT_ROOT}/lib64")
        SET(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT}/lib64")
    ELSEIF(EXISTS "${CUDA_TOOLKIT_ROOT}/lib")
        SET(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT}/lib")
    ELSE()
        MESSAGE(FATAL_ERROR "Could not find CUDA library directory")
    ENDIF()
    
    IF(EXISTS "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include")
        SET(CUDA_TOOLKIT_INCLUDE "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include")
    ELSE()
        SET(CUDA_TOOLKIT_INCLUDE "${CUDA_TOOLKIT_ROOT}/include")
    ENDIF()
    MESSAGE(STATUS "CUDA_TOOLKIT_INCLUDE: ${CUDA_TOOLKIT_INCLUDE}")

    MESSAGE(STATUS "CUDA_LIB_DIR: ${CUDA_LIB_DIR}")
    SET(CUDA_nvrtc_LIBRARY "${CUDA_LIB_DIR}/libnvrtc.so")
    
    # Set CUDA runtime libraries
    SET(CUDA_LIBRARIES "${CUDA_LIB_DIR}/libcudart.so")
    
    # Set CUDA architectures if not already set
    IF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        SET(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80" CACHE STRING "CUDA architectures to compile for")
    ENDIF()
    
ELSE(CMAKE_CUDA_COMPILER)
    MESSAGE("Found CUDA: FALSE")
    SET(PLUGIN_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
ENDIF(CMAKE_CUDA_COMPILER)

# Download and configure VkFFT first
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

IF(OPENCL_FOUND OR CMAKE_CUDA_COMPILER)
    include(ExternalProject)
    SET(VKFFT_VERSION "v1.3.4")
    SET(VKFFT_SOURCE_DIR "${CMAKE_BINARY_DIR}/vkfft-src")
    SET(VKFFT_INCLUDE_DIR "${VKFFT_SOURCE_DIR}/vkFFT")
    MESSAGE(STATUS "VKFFT_SOURCE_DIR: ${VKFFT_SOURCE_DIR}")
    MESSAGE(STATUS "VKFFT_INCLUDE_DIR: ${VKFFT_INCLUDE_DIR}")

    # Download VkFFT
    IF(NOT EXISTS "${VKFFT_SOURCE_DIR}")
        MESSAGE(STATUS "Downloading VkFFT ${VKFFT_VERSION}")
        EXECUTE_PROCESS(
            COMMAND git -c advice.detachedHead=false clone --branch ${VKFFT_VERSION} --depth 1 https://github.com/DTolm/VkFFT.git ${VKFFT_SOURCE_DIR}
            RESULT_VARIABLE GIT_RESULT
        )
        IF(NOT GIT_RESULT EQUAL 0)
            MESSAGE(FATAL_ERROR "Failed to clone VkFFT repository")
        ENDIF()
    ENDIF()

    # Add VkFFT include directory globally
    INCLUDE_DIRECTORIES(${VKFFT_INCLUDE_DIR})
ENDIF(OPENCL_FOUND OR CMAKE_CUDA_COMPILER)

# Build the implementations for different platforms
ADD_SUBDIRECTORY(platforms/reference)
ADD_SUBDIRECTORY(platforms/common)
IF(PLUGIN_BUILD_OPENCL_LIB)
    ADD_SUBDIRECTORY(platforms/opencl)
ENDIF(PLUGIN_BUILD_OPENCL_LIB)
IF(PLUGIN_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(PLUGIN_BUILD_CUDA_LIB)

# Build the Python API

FIND_PROGRAM(DOXYGEN_EXECUTABLE doxygen)
FIND_PROGRAM(PYTHON_EXECUTABLE python REQUIRED)
FIND_PROGRAM(SWIG_EXECUTABLE swig REQUIRED)
SET(PLUGIN_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
IF(PLUGIN_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(PLUGIN_BUILD_PYTHON_WRAPPERS)

# Build the documentation

ADD_SUBDIRECTORY(docs)

# Find required packages

SET(CMAKE_THREAD_PREFER_PTHREAD ON CACHE BOOL "Prefer pthread lib")
SET(THREADS_PREFER_PTHREAD_FLAG ON CACHE BOOL "Prefer pthread flag")
FIND_PACKAGE(Threads REQUIRED)
