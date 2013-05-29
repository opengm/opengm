# - this module looks for Matlab
# Defines:
# MATLAB_ROOT_DIR: path to Matlab's root directory
# MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
# MATLAB_LIBRARY_DIR : map path for mexFunction.map, ....
# MATLAB_LIBRARIES: required libraries: libmex, etc
# MATLAB_MEX_LIBRARY: path to libmex.lib
# MATLAB_MX_LIBRARY: path to libmx.lib
# MATLAB_ENG_LIBRARY: path to libeng.lib
#
## config

set(MATLAB_ROOT_DIR "" CACHE PATH "CPLEX root directory")
set(MATLAB_FOUND 0)

if(WIN32)
 #TODO
endif(WIN32)

if(UNIX)
  ## LINUX 64 BIT
  SET(MATLAB_MEXEXT "mexa64" CACHE STRING "Matlab MEX file extension")

  FIND_PATH(MATLAB_INCLUDE_DIR
    mex.h
    HINTS ${MATLAB_ROOT_DIR}/extern/include/
    PATHS ENV C_INCLUDE_PATH
          ENV C_PLUS_INCLUDE_PATH
          ENV INCLUDE_PATH
  ) 
  FIND_PATH(
    MATLAB_LIBRARY_DIR 
    mexFunction.map 
    HINTS ${MATLAB_ROOT_DIR}/extern/lib/glnxa64/
    ) 
  FIND_LIBRARY(MATLAB_MEX_LIBRARY
    libmex.so
    HINTS ${MATLAB_ROOT_DIR}/bin/glnxa64/
    PATHS ENV LIBRARY_PATH #unix
          ENV LD_LIBRARY_PATH #unix
  )
  FIND_LIBRARY(MATLAB_MX_LIBRARY
    libmx.so
    HINTS ${MATLAB_ROOT_DIR}/bin/glnxa64/
    PATHS ENV LIBRARY_PATH #unix
          ENV LD_LIBRARY_PATH #unix
  ) 
  FIND_LIBRARY(MATLAB_ENG_LIBRARY
    libeng.so
    HINTS ${MATLAB_ROOT_DIR}/bin/glnxa64/
    PATHS ENV LIBRARY_PATH #unix
          ENV LD_LIBRARY_PATH #unix
  ) 
 
endif(UNIX)

SET(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
  )

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND 1)
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_LIBRARY_DIR 
  MATLAB_MEXEXT
  MATLAB_FOUND
)
