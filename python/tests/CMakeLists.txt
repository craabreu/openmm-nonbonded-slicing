file(GLOB test_files ${CMAKE_CURRENT_SOURCE_DIR}/Test*.py)
foreach(file ${test_files})
    configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
endforeach(file ${test_files})
add_custom_target(PythonTest DEPENDS "${test_files}")
add_custom_command(
    TARGET PythonTest
    COMMAND "${PYTHON_EXECUTABLE}" -m pytest --verbose Test*.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
