find_program(SPHINX_EXECUTABLE sphinx-build)

file(GLOB STAGING_INPUT_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "_static/logo.png"
    "_static/logo_small.png"
    "_static/css/custom.css"
    "authors.rst"
    "changelog.rst"
    "conf.py"
    "contributing.rst"
    "index.rst"
    "overview.rst"
    "pythonapi/index.rst"
    "references.rst"
    "refs.bib"
    "theory.rst"
)

set(STAGING_OUTPUT_FILES "")
foreach(INIT_FILE ${STAGING_INPUT_FILES})
    set(infile "${CMAKE_CURRENT_SOURCE_DIR}/${INIT_FILE}")
    set(outfile "${CMAKE_CURRENT_BINARY_DIR}/${INIT_FILE}")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${infile}" "${outfile}"
        DEPENDS "${infile}"
        COMMENT "CMake-copying file ${infile} to ${outfile}")
    set(STAGING_OUTPUT_FILES ${STAGING_OUTPUT_FILES} "${outfile}")
endforeach(INIT_FILE ${STAGING_INPUT_FILES})

set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

add_custom_target(doc
    DEPENDS PythonInstall ${STAGING_OUTPUT_FILES}
)
add_custom_command(
    TARGET doc
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${SPHINX_EXECUTABLE} . ${SPHINX_HTML_DIR}
)
