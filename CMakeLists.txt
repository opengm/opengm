#--------------------------------------------------------------
# default build-type (release)
# (the next lines must be called bevore project(opengm2))
#--------------------------------------------------------------
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

#--------------------------------------------------------------
# OpenGM
#--------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
project(opengm2)
set (OPENGM_VERSION_MAJOR 2)
set (OPENGM_VERSION_MINOR 3)
set (OPENGM_VERSION_PATCH 1)

#--------------------------------------------------------------
# global headers
#--------------------------------------------------------------
file(GLOB_RECURSE headers include/*.hxx)
include_directories(include)

#--------------------------------------------------------------
# debug info
#--------------------------------------------------------------
#add_definitions(-DTRWS_DEBUG_OUTPUT)

#--------------------------------------------------------------
# warning level
#-------------------------------------------------------------- 
SET(WARNINGLEVEL "0" CACHE STRING "selected level for compiler warning from  0 (sloppy) to 4 (sadistic)") 
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
   # no warnings
   if(WARNINGLEVEL STREQUAL "0" )
     message(STATUS "--------------------------------------------------------------------------")
     message(STATUS "WARNING: Compiler warnings are very sloppy -> increase CMake-WARNINGLEVEL")
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
   # SADISTIC -> all warnings become errors
   if(WARNINGLEVEL STREQUAL "4" )
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++98 -pedantic -Wno-long-long -Wextra -Werror")
   endif()
elseif(MSVC)
   add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif()


#--------------------------------------------------------------
# options
#--------------------------------------------------------------
OPTION(BUILD_EXAMPLES "Build Examples" ON)
OPTION(BUILD_COMMANDLINE "Build Commandline" OFF)
OPTION(WITH_AD3 "Include AD3" OFF)
OPTION(WITH_CPLEX "Include CPLEX" OFF)
OPTION(WITH_GUROBI "Include GUROBI" OFF)
OPTION(WITH_BOOST "Include BOOST" OFF)
OPTION(WITH_HDF5 "Include HDF5" OFF)
OPTION(WITH_TRWS "Include wrapper for TRWS code" OFF)
OPTION(WITH_GCO "Include wrapper for GCO code" OFF)
OPTION(WITH_MRF "Include wrapper for MRF code" OFF)
OPTION(WITH_QPBO "Include wrapper for QPBO code" OFF)
OPTION(WITH_FASTPD "Include wrapper for FastPD code" OFF)
OPTION(WITH_MAXFLOW "Include wrapper for MAXFLOW code" OFF)
OPTION(WITH_MAXFLOW_IBFS "Include wrapper for MAXFLOW-IBFScode" OFF)
OPTION(WITH_LIBDAI "Include wrapper for LIBDAI code" OFF)
OPTION(WITH_DAOOPT "Include wrapper for DAOOPT code" OFF)
OPTION(WITH_MPLP "Include wrapper for MPLP code" OFF)
OPTION(WITH_OPENMP "Include OpenMP" OFF)
OPTION(WITH_CONICBUNDLE "Include ConicBundleLibrary" OFF)
OPTION(WITH_MATLAB "Activate MatLab-Code" OFF)
OPTION(BUILD_CONVERTER "Build several converters" OFF)
OPTION(BUILD_PYTHON_WRAPPER "Build python wrapper" OFF)
OPTION(BUILD_MATLAB_WRAPPER "Build matlab wrapper" OFF)
###Grante needs C++11. Since we have not tested OpenGM under this standard yet, using Grante is realy experimental!!!
###OPTION(WITH_GRANTE "Include wrapper for grante" OFF)

#--------------------------------------------------------------
# Cplex
#--------------------------------------------------------------
if(WITH_CPLEX)
   message(STATUS "build with CPLEX interface")
   find_package(CPLEX REQUIRED)
   add_definitions(-DWITH_CPLEX)
   include_directories(${CPLEX_INCLUDE_DIRS})
   # add_definitions(-DILOUSESTL) redundant according to http://www-01.ibm.com/support/docview.wss?uid=swg21399983
   add_definitions(-DIL_STD)
else()
   message(STATUS "build without CPLEX interface")
endif()


#--------------------------------------------------------------
# AD3
#--------------------------------------------------------------
if(WITH_AD3)
   #message(STATUS "build with AD3 interface")
   #find_package(AD3 REQUIRED)
   #add_definitions(-DWITH_AD3)
   #include_directories(${AD3_INCLUDE_DIR})


  message(STATUS "build with external inference algorithm TRWS")
  SET(AD3_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/AD3-patched" CACHE STRING "TRWS patched source code directory")
  add_definitions(-DWITH_AD3)
  include_directories(${AD3_PATCHEDSRCDIR})



else()
   message(STATUS "build without AD3 interface")
endif()

#--------------------------------------------------------------
# Gurobi
#--------------------------------------------------------------
if(WITH_GUROBI)
   message(STATUS "build with GUROBI interface")
   find_package(GUROBI REQUIRED)
   add_definitions(-DWITH_GUROBI)
   include_directories(${GUROBI_INCLUDE_DIRS})
else()
   message(STATUS "build without GUROBI interface")
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
  find_package(MATLAB REQUIRED)
  message(STATUS "Matlab include dir: ${MATLAB_INCLUDE_DIR}")
  message(STATUS "Matlab libmex: ${MATLAB_MEX_LIBRARY}")
  message(STATUS "Matlab libeng: ${MATLAB_ENG_LIBRARY}")
  message(STATUS "Matlab libmx:  ${MATLAB_MX_LIBRARY}")

  #include_directories(${MATLAB_INCLUDE_DIR})
  #add_definitions(-DWITH_MATLAB)
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
# GCO
#--------------------------------------------------------------
if(WITH_GCO)
   message(STATUS "build with external inference algorithm GCO")
   SET(GCO_LABEL_VALUE "int" CACHE STRING "selected label type for GCO library")
   add_definitions(-DGCOLABELVALUE=${GCO_LABEL_VALUE})
   SET(GCO_ENERGY_VALUE "double" CACHE STRING "selected energy type for GCO library")
   add_definitions(-DGCOENERGYVALUE=${GCO_ENERGY_VALUE})
   SET(GCO_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/GCO-v3.0.src-patched" CACHE STRING "GCO patched source code directory")
   add_definitions(-DWITH_GCO)
   include_directories(${GCO_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm GCO")
endif(WITH_GCO)

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
# FastPD
#--------------------------------------------------------------
if(WITH_FASTPD)
   message(STATUS "build with external inference algorithm FastPD")
   SET(FASTPD_URL "" CACHE STRING "URL for downloading FastPD (Registration required at http://www.csd.uoc.gr/~komod/FastPD/)")
   SET(FASTPD_ENERGY_VALUE "double" CACHE STRING "selected energy type for FASTPD library")
   add_definitions(-DFASTPDENERGYVALUE=${FASTPD_ENERGY_VALUE})
   SET(FASTPD_LABEL_VALUE "size_t" CACHE STRING "selected label type for FASTPD library")
   add_definitions(-DFASTPDLABELVALUE=${FASTPD_LABEL_VALUE})
   SET(FASTPD_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/FastPD.src-patched" CACHE STRING "FastPD patched source code directory")
   add_definitions(-DWITH_FASTPD)
   include_directories(${FASTPD_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm FastPD")
endif(WITH_FASTPD)

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
# grante Library
#-------------------------------------------------------------
if(WITH_GRANTE)
    if(WITH_BOOST)
       message(STATUS "build with external inference algorithm grante")
       SET(GRANTE_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/grante-v1.0.src-patched/" CACHE STRING "Grante source code directory")
       add_definitions(-DWITH_GRANTE)
       include_directories(${GRANTE_PATCHEDSRCDIR})
       set(Boost_USE_STATIC_LIBS ON)
       find_package(Boost 1.47 COMPONENTS serialization REQUIRED)
    else(WITH_BOOST)
        message(SEND_ERROR "Grante requires Boost! Enable WITH_BOOST in CMake.")
    endif(WITH_BOOST)

else()
   message(STATUS "build without external inference library grante")
endif(WITH_GRANTE)

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
   
  # Add defines for libdai/include/dai/util.h
  if(WIN32)
     add_definitions(-DWINDOWS)
  else(CYGWIN)
     add_definitions(-DCYGWIN)
  else(APPLE)
     add_definitions(-DMACOSX)
  endif()

else()
   message(STATUS "build without external inference algorithms from  LibDai")
endif(WITH_LIBDAI)

#--------------------------------------------------------------
# DAOOPT
#--------------------------------------------------------------
if(WITH_DAOOPT)
   if(WITH_BOOST)
      message(STATUS "build with external inference algorithm DAOOPT")
      SET(DAOOPT_SRCDIR "${PROJECT_SOURCE_DIR}/src/external/DAOOPT.src/" CACHE STRING "DAOOPT source code directory")
      add_definitions(-DWITH_DAOOPT)
      add_definitions(-DWITH_OPENGM)
      include_directories(${DAOOPT_SRCDIR}/include
                          ${DAOOPT_SRCDIR}/lib
      )
      set(Boost_USE_MULTITHREADED OFF)
      
      # To select the two master modes
      option(DAOOPT_WORKER "Force worker binary" OFF)
      option(DAOOPT_MASTER_STATIC "Force static master binary" OFF)
      option(DAOOPT_MASTER_DYNAMIC "Force dynamic master binary" OFF)
        
      # To enable static linking of the final daoopt binary
      option(DAOOPT_LINK_STATIC "Link binary statically" OFF)
      
      # Add defines for DAOOPT
      if(WIN32)
         add_definitions(-DWINDOWS)
      else()
         add_definitions(-DLINUX)
      endif()
      
      if(DAOOPT_LINK_STATIC)
         set(CMAKE_EXE_LINKER_FLAGS "-static")
      endif()
      
      # Optional forced version: static over dynamic master, worker over both
      if(DAOOPT_WORKER)
         add_definitions(-DNOTHREADS)
      else()
         if(DAOOPT_MASTER_STATIC)
            add_definitions(-DPARALLEL_STATIC)
         endif()
         if(DAOOPT_MASTER_DYNAMIC AND NOT DAOOPT_MASTER_STATIC)
            add_definitions(-DPARALLEL_DYNAMIC)
            find_package(Threads)  # includes pthread
            find_package( Boost REQUIRED COMPONENTS system )
         endif()
      endif()
      find_package( Boost REQUIRED COMPONENTS program_options thread )
      include_directories( ${Boost_INCLUDE_DIRS} )
   else(WITH_BOOST)
      message(SEND_ERROR "DAOOPT requires Boost! Enable WITH_BOOST in CMake.")
   endif(WITH_BOOST)
else()
   message(STATUS "build without external inference algorithm DAOOPT")
endif(WITH_DAOOPT)

#--------------------------------------------------------------
# MPLP
#--------------------------------------------------------------
if(WITH_MPLP)
   message(STATUS "build with external inference algorithm MPLP")
   SET(MPLP_PATCHEDSRCDIR "${PROJECT_SOURCE_DIR}/src/external/mplp_ver2.src-patched/" CACHE STRING "MPLP patched source code directory")
   add_definitions(-DWITH_MPLP)
   include_directories(${MPLP_PATCHEDSRCDIR})
else()
   message(STATUS "build without external inference algorithm MPLP")
endif(WITH_MPLP)

#--------------------------------------------------------------
# thread lib
#--------------------------------------------------------------
find_package(Threads)

#--------------------------------------------------------------
# rt lib
#--------------------------------------------------------------
if(UNIX AND NOT APPLE)
  find_library(RT rt) 
  set(LINK_RT true) 
  message(STATUS "Linking to RT is enabled")
else()
  set(LINK_RT false) 
  message(STATUS "Linking to RT is diabled")
endif()

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
   # find numpy
   find_package(NUMPY)
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
      "from __future__ import print_function; from distutils.sysconfig import get_python_lib; print(get_python_lib(1))"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
      FILE(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} OPENGM_PYTHON_MODULE_INSTALL_DIR)
   ENDIF()
   SET(OPENGM_PYTHON_MODULE_INSTALL_DIR ${OPENGM_PYTHON_MODULE_INSTALL_DIR}
   CACHE PATH "where to install the OpenGM Python package" FORCE)
   # this is the install path relative to CMAKE_INSTALL_PREFIX,
   # use this in INSTALL() commands to get packaging right
   FILE(RELATIVE_PATH OPENGM_PYTHON_MODULE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${OPENGM_PYTHON_MODULE_INSTALL_DIR})

   execute_process( COMMAND python -c 
      "from __future__ import print_function; from distutils.sysconfig import get_python_lib; print(get_python_lib())" 
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE 
   ) 

   #find_path(PYTHON_MODULE_INSTALL )   
   install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/interfaces/python/opengm" DESTINATION "${OPENGM_PYTHON_MODULE_INSTALL_DIR}" 
   PATTERN ".py" 
   PATTERN ".git"  EXCLUDE 
   PATTERN  ".txt" EXCLUDE
   PATTERN  ".hxx" EXCLUDE )
endif()



#--------------------------------------------------------------
# package with cpack
# - this is just a starting point
#   
#--------------------------------------------------------------

#SET(CPACK_GENERATOR "DEB")
#SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Thorsten Beier") #required




#set(CPACK_PACKAGE_VERSION_MAJOR "${OPENGM_VERSION_MAJOR}")
#set(CPACK_PACKAGE_VERSION_MINOR "${OPENGM_VERSION_MINOR}")
#set(CPACK_PACKAGE_VERSION_PATCH "${OPENGM_VERSION_PATCH}")

#INCLUDE(CPack)
