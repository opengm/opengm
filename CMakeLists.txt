#--------------------------------------------------------------
# OpenGM
#--------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
project(opengm2)
set (OPENGM_VERSION_MAJOR 2)
set (OPENGM_VERSION_MINOR 0)
set (OPENGM_VERSION_PATCH 2)

#--------------------------------------------------------------
# global headers
#--------------------------------------------------------------
file(GLOB_RECURSE headers include/*.hxx)
include_directories(include)

#--------------------------------------------------------------
# warning level
#-------------------------------------------------------------- 
SET(WARNINGLEVEL "0" CACHE STRING "selected level for compiler warning from  0 (sloppy) to 4 (sadistic)") 
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
   # no warnings
   if(WARNINGLEVEL STREQUAL "0" )
     message(STATUS "--------------------------------------------------------------------------")
     message(STATUS "WARRNING: Compiler warnings are very sloppy -> increase CMake-WARNINGLEVEL")
     message(STATUS "--------------------------------------------------------------------------")
   endif()
   # with warning
   if(WARNINGLEVEL STREQUAL "1" )
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++98")
   endif()
   # PEDANTIC -> a lot of warnings
   if(WARNINGLEVEL STREQUAL "2" )
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++98 -pedantic -Wno-long-long ")
   endif() 
   # VERY PEDANTIC -> very lot of warnings
   if(WARNINGLEVEL STREQUAL "3" )
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++98 -pedantic -Wno-long-long -Wextra")
   endif()
   # SADISTIC -> all warrnings become errors
   if(WARNINGLEVEL STREQUAL "4" )
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall-std=c++98 -pedantic -Wno-long-long -Wextra -Werror")
   endif()
elseif(MSVC)
   add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif()

#--------------------------------------------------------------
# options
#--------------------------------------------------------------
OPTION(BUILD_EXAMPLES "Build Examples" ON)
OPTION(BUILD_COMMANDLINE "Build Commandline" OFF)
OPTION(WITH_CPLEX "Include CPLEX" OFF)
OPTION(WITH_BOOST "Include BOOST" OFF)
OPTION(WITH_HDF5 "Include HDF5" OFF)
OPTION(WITH_TRWS "Include wrapper for TRWS code" OFF)
OPTION(WITH_MRF "Include wrapper for MRF code" OFF)
OPTION(WITH_QPBO "Include wrapper for QPBO code" OFF)
OPTION(WITH_MAXFLOW "Include wrapper for MAXFLOW code" OFF)
OPTION(WITH_MAXFLOW_IBFS "Include wrapper for MAXFLOW-IBFScode" OFF)
OPTION(WITH_LIBDAI "Include wrapper for LIBDAI code" OFF)
OPTION(WITH_OPENMP "Include OpenMP" OFF)
OPTION(WITH_CONICBUNDLE "Include ConicBundleLibrary" OFF)
OPTION(WITH_MATLAB "Activate MatLab-Code" OFF)
OPTION(WITH_CONVERTER "Build Converter OpenGm1->OpenGm2, requires Boost 1.47!" OFF)
OPTION(BUILD_PYTHON_WRAPPER "build python wrapper" OFF)

#--------------------------------------------------------------
# Cplex
#--------------------------------------------------------------
if(WITH_CPLEX)
   message(STATUS "build with CPLEX interface")
   find_package(CPLEX REQUIRED)
   add_definitions(-DWITH_CPLEX)
   include_directories(${CPLEX_INCLUDE_DIR})
   include_directories(${CONCERT_INCLUDE_DIR})
   add_definitions(-DILOUSESTL)
   add_definitions(-DIL_STD)
else()
   message(STATUS "build without CPLEX interface")
endif()

#--------------------------------------------------------------
# Boost
#--------------------------------------------------------------
if(WITH_BOOST)
   message(STATUS "build with Boost")
   find_package(Boost 1.47 REQUIRED)
   add_definitions(-DWITH_BOOST)
   include_directories(${Boost_INCLUDE_DIR})
else()
   message(STATUS "build without Boost")
endif()

#--------------------------------------------------------------
# HDF5
#--------------------------------------------------------------
if(WITH_HDF5)
   message(STATUS "build with HDF5 support")
   find_package(HDF5 REQUIRED)
   include_directories(${HDF5_INCLUDE_DIR})
   add_definitions(${HDF5_CPPFLAGS})
   add_definitions(-DWITH_HDF5)
else()
   message(STATUS "build without HDF5")
endif()

#--------------------------------------------------------------
# MATLAB
#--------------------------------------------------------------
if(WITH_MATLAB)
  message(STATUS "build with MatLab support")
  find_package(Matlab REQUIRED)
  include_directories(${MATLAB_INCLUDE_DIR})
  add_definitions(-DWITH_MATLAB)
endif(WITH_MATLAB)

#--------------------------------------------------------------
# TRWS
#--------------------------------------------------------------
if(WITH_TRWS)
   message(STATUS "build with external inference algorithm TRWS")
   SET(TRWS_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/TRWS-v1.3.src-patched" CACHE STRING "TRWS patched source code directory")
   add_definitions(-DWITH_TRWS)
   include_directories(${TRWS_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm TRWS")
endif(WITH_TRWS)

#--------------------------------------------------------------
# MRF
#--------------------------------------------------------------
if(WITH_MRF)
   message(STATUS "build with external inference algorithm MRF")
   SET(MRF_LABEL_VALUE "int" CACHE STRING "selected label type for MRF library")
   add_definitions(-DMRFLABELVALUE=${MRF_LABEL_VALUE})
   SET(MRF_ENERGY_VALUE "double" CACHE STRING "selected energy type for MRF library")
   add_definitions(-DMRFENERGYVALUE=${MRF_ENERGY_VALUE})
   SET(MRF_COST_VALUE "double" CACHE STRING "selected cost type for MRF library")
   add_definitions(-DMRFCOSTVALUE=${MRF_COST_VALUE})
   SET(MRF_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/MRF-v2.1.src-patched" CACHE STRING "MRF patched source code directory") 
 
   add_definitions(-DWITH_MRF)
   include_directories(${MRF_PATCHEDSRCDIR})

   set(bitness 32)
   if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      MESSAGE (STATUS "Detected a 64 Bit Machine")
      set(bitness 64)
      add_definitions(-DUSE_64_BIT_PTR_CAST)
   else()
      MESSAGE (STATUS "Detected a 32 Bit Machine")
   endif()
  
   #TODO add 64 bit check here!
   #if(64Bit???)
     #add_definitions(-DUSE_64_BIT_PTR_CAST)
   #endif(64Bit???)      
else()
   message(STATUS "build without external inference algorithm MRF")
endif(WITH_MRF)

#--------------------------------------------------------------
# QPBO
#--------------------------------------------------------------
if(WITH_QPBO)
   message(STATUS "build with external inference algorithm QPBO")
   SET(QPBO_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/QPBO-v1.3.src-patched" CACHE STRING "QPBO patched source code directory")
   add_definitions(-DWITH_QPBO)
   include_directories(${QPBO_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm QPBO")
endif(WITH_QPBO)

#--------------------------------------------------------------
# MaxFlow-IBFS
#--------------------------------------------------------------
if(WITH_MAXFLOW_IBFS)
   message(STATUS "build with external inference algorithm MaxFlow-IBFS")
   SET(MAXFLOW_IBFS_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/ibfs.src-patched/" CACHE STRING "MAXFLOW-IBFS source code directory")
   add_definitions(-DWITH_MAXFLOW_IBFS)
   include_directories(${MAXFLOW_IBFS_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm MaxFlow-IBFS")
endif(WITH_MAXFLOW_IBFS)

#--------------------------------------------------------------
# MaxFlow
#--------------------------------------------------------------
if(WITH_MAXFLOW)
   message(STATUS "build with external inference algorithm MaxFlow")
   SET(MAXFLOW_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/MaxFlow-v3.02.src-patched/" CACHE STRING "MAXFLOW patched source code directory")
   add_definitions(-DWITH_MAXFLOW)
   include_directories(${MAXFLOW_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm MaxFlow")
endif(WITH_MAXFLOW)


#-------------------------------------------------------------
# ConicBundle Library
#-------------------------------------------------------------
if(WITH_CONICBUNDLE) 
    IF(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/external/ConicBundle-v0.3.11.src-patched")
        MESSAGE ("ConicBundle not installed, run make externalLibs first and configure again")
        SET(WITH_CONICBUNDLE OFF)
    ELSE(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/external/ConicBundle-v0.3.11.src-patched")
        message(STATUS "build with ConicBundle-Library") 
        find_package(CONICBUNDLE REQUIRED)
        include_directories(${CONICBUNDLE_INCLUDE_DIR})
        add_definitions(-DWITH_CONICBUNDLE) 
        add_definitions(-DWITH_BUNDLE)
    ENDIF(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/external/ConicBundle-v0.3.11.src-patched")
else()
   message(STATUS "build without inference algorithm DDBundle")
endif(WITH_CONICBUNDLE)

#-------------------------------------------------------------
# OpenMP
#-------------------------------------------------------------
if(WITH_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  message(STATUS "build with OpenMP") 
  #SET(OPENMP_INCLUDE_DIR "" CACHE STRING "OpenMP include dir")
  #include_directories(${OPENMP_INCLUDE_DIR})
  add_definitions(-DWITH_OPENMP)
else()
   message(STATUS "build without openMP -> multithreaded options disabled")
endif(WITH_OPENMP)

#--------------------------------------------------------------
# LIB DAI
#--------------------------------------------------------------
if(WITH_LIBDAI)
   find_package(LIBDAI  REQUIRED)
   find_package(GMP  REQUIRED)
   find_package(GMPXX  REQUIRED)
   include_directories(${LIBDAI_INCLUDE_DIR})
   include_directories(${GMP_INCLUDE_DIR})
   include_directories(${GMPXX_INCLUDE_DIR})
   add_definitions(-DWITH_LIBDAI)
else()
   message(STATUS "build without external inference algorithms from  LibDai")
endif(WITH_LIBDAI)

#--------------------------------------------------------------
# PYTHON WRAPPER
#--------------------------------------------------------------
if(BUILD_PYTHON_WRAPPER)
    find_package(Boost COMPONENTS python REQUIRED)
    find_package(HDF5 REQUIRED)
    find_package (PythonLibs REQUIRED)
    add_definitions(-DWITH_BOOST)
    add_definitions(-DWITH_HDF5)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${Boost_PYTHON_INCLUDE_DIR}
)
endif()

#--------------------------------------------------------------
# thread lib
#--------------------------------------------------------------
find_package(Threads)

#--------------------------------------------------------------
# source directory
#--------------------------------------------------------------
add_subdirectory(src)

#--------------------------------------------------------------
# testing with CTest
#--------------------------------------------------------------
INCLUDE(CTest)
ENABLE_TESTING()
if(BUILD_TESTING)
   add_subdirectory(src/unittest)
endif()

#--------------------------------------------------------------
# WordLength
#--------------------------------------------------------------

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
   MESSAGE (STATUS "Detected a 64 Bit Machine")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
   MESSAGE (STATUS "Detected a 32 Bit Machine")
endif()
set(OPENGM_WORD_LENGTH ${CMAKE_SIZEOF_VOID_P})


#--------------------------------------------------------------
# TypeChecking
#--------------------------------------------------------------
INCLUDE (${CMAKE_ROOT}/Modules/CheckTypeSize.cmake) 
set(listTypes "char"
"signed char"
"unsigned char"
"wchar_t"
"short"
"unsigned short"
"int"
"unsigned int"
"long"
"unsigned long"
"long long"
"unsigned long long"
"float"
"double"
"long double" )

set(listTypeNames CHAR  
SIGNED_CHAR  
UNSIGNED_CHAR  
WCHAR_T  
SHORT  
UNSIGNED_SHORT  
INT  
UNSIGNED_INT  
LONG  
UNSIGNED_LONG  
LONG_LONG  
UNSIGNED_LONG_LONG  
FLOAT  
DOUBLE  
LONG_DOUBLE  )

MACRO (OPENGM_CHECK_TYPE_SIZE type var)
   SET (aType ${type})
   SET (sizeVar  OPENGM_SIZE_OF_${var})
   CHECK_TYPE_SIZE (${aType} ${sizeVar})
   IF(NOT ${sizeVar} )
      SET (${sizeVar} 0 CACHE INTERNAL "SizeOf for ${sizeVar}")
      MESSAGE (STATUS "Type ${aType} was NOT Found")
      SET(OPENGM_NO_${var} ON )
      MESSAGE (STATUS "Set  ${noTypeVar} ")
   ELSE(NOT ${sizeVar})
      #MESSAGE (STATUS "Size of ${aType} is ${${sizeVar}}")
   ENDIF(NOT ${sizeVar})
ENDMACRO (OPENGM_CHECK_TYPE_SIZE)

list(LENGTH listTypes sizelistTypes)
math(EXPR sizeOfList "${sizelistTypes} - 1")

foreach(val RANGE ${sizeOfList})
  list(GET listTypes ${val} valTypes)
  list(GET listTypeNames ${val} valTypeNames)
  #message(STATUS "Check Type : ${valTypes}")
  OPENGM_CHECK_TYPE_SIZE( ${valTypes}  ${valTypeNames} )
endforeach()

#--------------------------------------------------------------
# doxygen documentation
#--------------------------------------------------------------
find_package(Doxygen)
if(DOXYGEN_FOUND)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
   add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

#--------------------------------------------------------------
# install
#--------------------------------------------------------------
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/opengm" DESTINATION include PATTERN ".hxx" PATTERN ".git" EXCLUDE)
#hack to make "make install" work even if the documentation has not been installed!
IF(IS_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc/html" )
   # do nothing
else()
   file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc/html")
endif()
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc/html"       DESTINATION doc/opengm  PATTERN ".git" EXCLUDE PATTERN ".cmake" EXCLUDE)

#--------------------------------------------------------------
# test and install opengm python
#--------------------------------------------------------------
if(BUILD_PYTHON_WRAPPER)
   #find python
   FIND_PACKAGE(PythonInterp)
   #find nose
   execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import nose"  RESULT_VARIABLE PYTHON_NOSETESTS_NOT_FOUND)
   #add python unit test
   IF(NOT PYTHON_NOSETESTS_NOT_FOUND)
      MESSAGE(STATUS "Searching for Python nosetests: ok")
      add_custom_target( test-python-wrapper nosetests 
      #add_custom_target( test-python-wrapper ALL nosetests 
        --all-modules           
        --traverse-namespace         
        --cover-tests          
        --with-doctest          
        -v WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/interfaces/python" 
        --where "${CMAKE_CURRENT_BINARY_DIR}/src/interfaces/python" 
      )
      add_dependencies(test-python-wrapper _opengmcore )
      add_dependencies(test-python-wrapper _inference )
      add_dependencies(test-python-wrapper _hdf5 )
   ELSE()
      MESSAGE(STATUS "Could NOT find Python nosetests ('import nose' failed)")
   ENDIF()


   #find default install directory for Python modules (usually PYTHONDIR/Lib/site-packages)
   IF(NOT DEFINED OPENGM_PYTHON_MODULE_INSTALL_DIR OR OPENGM_PYTHON_MODULE_INSTALL_DIR MATCHES "^$")
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c 
      "from distutils.sysconfig import *; print get_python_lib(1)"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
      FILE(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} OPENGM_PYTHON_MODULE_INSTALL_DIR)
   ENDIF()
   SET(OPENGM_PYTHON_MODULE_INSTALL_DIR ${OPENGM_PYTHON_MODULE_INSTALL_DIR}
   CACHE PATH "where to install the OpenGM Python package" FORCE)
   # this is the install path relative to CMAKE_INSTALL_PREFIX,
   # use this in INSTALL() commands to get packaging right
   FILE(RELATIVE_PATH OPENGM_PYTHON_MODULE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${OPENGM_PYTHON_MODULE_INSTALL_DIR})

   execute_process( COMMAND python -c 
      "from distutils.sysconfig import get_python_lib; print get_python_lib()" 
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE 
   ) 

   #find_path(PYTHON_MODULE_INSTALL )   
   install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/interfaces/python/opengm" DESTINATION "${OPENGM_PYTHON_MODULE_INSTALL_DIR}" 
   PATTERN ".py" 
   PATTERN ".git"  EXCLUDE 
   PATTERN  ".txt" EXCLUDE
   PATTERN  ".hxx" EXCLUDE )
endif()
