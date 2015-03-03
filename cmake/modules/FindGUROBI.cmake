#### Taken from http://www.openflipper.org/svnrepo/CoMISo/trunk/CoMISo/cmake/FindGUROBI.cmake


# - Try to find GUROBI
# Once done this will define
# GUROBI_FOUND - System has Gurobi
# GUROBI_INCLUDE_DIRS - The Gurobi include directories
# GUROBI_LIBRARIES - The libraries needed to use Gurobi

if (GUROBI_INCLUDE_DIR)
  # in cache already
  set(GUROBI_FOUND TRUE)
  set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}" )
  set(GUROBI_LIBRARIES "${GUROBI_LIBRARY};${GUROBI_CXX_LIBRARY}" )
else (GUROBI_INCLUDE_DIR)

find_path(GUROBI_INCLUDE_DIR
          NAMES gurobi_c++.h
          PATHS "$ENV{GUROBI_HOME}/include"
                  "/Library/gurobi502/mac64/include"
                 "C:\\libs\\gurobi502\\include"
          )

find_library( GUROBI_LIBRARY
              NAMES gurobi
gurobi45
gurobi46
        gurobi50
        gurobi51
        gurobi52
        gurobi55
        gurobi60
              PATHS "$ENV{GUROBI_HOME}/lib"
                    "/Library/gurobi502/mac64/lib"
                    "C:\\libs\\gurobi502\\lib"
              )

if (MSVC)
  STRING(REGEX REPLACE "/VC/bin/.*" "" VISUAL_STUDIO_PATH ${CMAKE_C_COMPILER})

  STRING(REGEX MATCH "Studio [0-9]+" VISUAL_STUDIO_VERSION ${VISUAL_STUDIO_PATH})
  STRING(REGEX REPLACE "Studio " "" VISUAL_STUDIO_VERSION ${VISUAL_STUDIO_VERSION})

  if (VISUAL_STUDIO_VERSION STREQUAL "10")
    set (VISUAL_STUDIO_YEAR "2010")
  elseif (VISUAL_STUDIO_VERSION STREQUAL "11")
    set (VISUAL_STUDIO_YEAR "2012")
  elseif (VISUAL_STUDIO_VERSION STREQUAL "12")
    set (VISUAL_STUDIO_YEAR "2013")
  else ()
    message(FATAL_ERROR "Unsupported compiler version: ${VISUAL_STUDIO_VERSION}")
  endif ()

  set (GUROBI_LIB_NAME gurobi_c++md${VISUAL_STUDIO_YEAR})
else ()
  set (GUROBI_LIB_NAME gurobi_c++)
endif ()


find_library( GUROBI_CXX_LIBRARY
              NAMES ${GUROBI_LIB_NAME}
              PATHS "$ENV{GUROBI_HOME}/lib"
                    "/Library/gurobi502/mac64/lib"
                    "C:\\libs\\gurobi502\\lib"
              )

set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}" )
set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}" )

# use c++ headers as default
# set(GUROBI_COMPILER_FLAGS "-DIL_STD" CACHE STRING "Gurobi Compiler Flags")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GUROBI DEFAULT_MSG
                                  GUROBI_LIBRARY GUROBI_CXX_LIBRARY GUROBI_INCLUDE_DIR)

mark_as_advanced(GUROBI_INCLUDE_DIR GUROBI_LIBRARY GUROBI_CXX_LIBRARY)

endif(GUROBI_INCLUDE_DIR)
