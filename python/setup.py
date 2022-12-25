from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
nonbondedslicing_header_dir = '@PLUGIN_HEADER_DIR@'
nonbondedslicing_library_dir = '@PLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

os.environ['CC'] = '@CMAKE_C_COMPILER@'
os.environ['CXX'] = '@CMAKE_CXX_COMPILER@'

extension = Extension(
    name='_nonbondedslicing',
    sources=['NonbondedSlicingWrapper.cpp'],
    libraries=['OpenMM', 'NonbondedSlicing'],
    include_dirs=[os.path.join(openmm_dir, 'include'), nonbondedslicing_header_dir],
    library_dirs=[os.path.join(openmm_dir, 'lib'), nonbondedslicing_library_dir],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='nonbondedslicing',
    version='1.0',
    py_modules=['nonbondedslicing'],
    ext_modules=[extension],
)
