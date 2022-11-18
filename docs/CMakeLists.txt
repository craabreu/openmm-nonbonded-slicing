find_program(SPHINX_EXECUTABLE sphinx-build)
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
include(CMakePrintHelpers)
cmake_print_variables(SPHINX_HTML_DIR)
cmake_print_variables(CMAKE_CURRENT_BINARY_DIR)
cmake_print_variables(CMAKE_BINARY_DIR)
cmake_print_variables(CMAKE_CURRENT_SOURCE_DIR)
file(GLOB DOC_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "authors.rst"
    "changelog.rst"
    "conf.py"
    "contributing.rst"
    "index.rst"
    "overview.rst"
    "pythonapi/index.rst"
    "pythonapi/SlicedPmeForce.rst"
    "references.rst"
    "refs.bib"
)
add_custom_target(doc
    DEPENDS PythonInstall ${DOC_FILES}
)
add_custom_command(
    TARGET doc
    COMMAND ${SPHINX_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ ${SPHINX_HTML_DIR}/
)
