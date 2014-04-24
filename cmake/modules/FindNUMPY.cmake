# Find the Python NumPy package
# PYTHON_NUMPY_INCLUDE_DIR
# NUMPY_FOUND
# will be set by this script


if(NUMPY_FIND_QUIETLY)
  find_package(PythonInterp)
else()
  find_package(PythonInterp QUIET)
  set(_numpy_out 1)
endif()

if (PYTHON_EXECUTABLE)
  # write a python script that finds the numpy path
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/FindNumpyPath.py
      "try: import numpy; print numpy.get_include()\nexcept:pass\n")

  # execute the find script
  exec_program("${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_BINARY_DIR}
    ARGS "FindNumpyPath.py"
    OUTPUT_VARIABLE NUMPY_PATH)
elseif(_numpy_out)
  message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
  "${NUMPY_PATH}"
  "${PYTHON_INCLUDE_PATH}"
  /usr/include/python2.7/
  /usr/include/python2.6/
  /usr/include/python2.5/
  /usr/include/python2.4/)

if(PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(PYTHON_NUMPY_INCLUDE_DIR)

#include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( NUMPY DEFAULT_MSG PYTHON_NUMPY_INCLUDE_DIR)