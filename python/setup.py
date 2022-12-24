from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
pmeslicing_header_dir = '@PMESLICING_HEADER_DIR@'
pmeslicing_library_dir = '@PMESLICING_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_pmeslicing',
                      sources=['PmeSlicingWrapper.cpp'],
                      libraries=['OpenMM', 'NonbondedSlicing'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), pmeslicing_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), pmeslicing_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='pmeslicing',
      version='1.0',
      py_modules=['pmeslicing'],
      ext_modules=[extension],
     )
