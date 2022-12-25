PYTHON_PATH=$(which python)
CONDA_PREFIX=${PYTHON_PATH%"/bin/python"}
echo "CONDA_PREFIX=${CONDA_PREFIX}"
mkdir build
cd build
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
-DOPENMM_DIR=${CONDA_PREFIX} \
-DPLUGIN_BUILD_OPENCL_LIB=OFF \
-DPLUGIN_BUILD_CUDA_LIB=OFF
make
make install
make PythonInstall
