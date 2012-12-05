
#--------------------------------------------------------------
# Find packages
#--------------------------------------------------------------
find_package(Boost COMPONENTS python REQUIRED)
find_package (PythonLibs REQUIRED)
if(WITH_HDF5)
   message(STATUS "build opengm python with HDF5 support")
   find_package(HDF5 REQUIRED)
   include_directories(${HDF5_INCLUDE_DIR})
   add_definitions(${HDF5_CPPFLAGS})
   add_definitions(-DWITH_HDF5)
else()
   message(STATUS "build opengm python without HDF5")
endif()
#--------------------------------------------------------------
# Include Directories
#--------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
	 ${PYTHON_INCLUDE_DIRS}
	 ${PYTHON_NUMPY_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${Boost_PYTHON_INCLUDE_DIR}
)





#--------------------------------------------------------------
# Add opengmcore library
#--------------------------------------------------------------
add_library(_inference SHARED  
inference.cpp
pyInference.cxx
pyIcm.cxx
pyGraphcut.cxx
pyMpBased.cxx
pyGibbs.cxx
pyBruteforce.cxx
pyLazyflipper.cxx
pyAe.cxx
pyAbSwap.cxx
pyDynp.cxx
pyCplex.cxx
external/pyLibdai.cxx
)





#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries(_inference ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARIES} rt)
IF(WITH_MAXFLOW)
   target_link_libraries(_inference  external-library-maxflow)
endif(WITH_MAXFLOW)
set_target_properties(_inference PROPERTIES PREFIX "")

if(WITH_LIBDAI)
   target_link_libraries(_inference ${LIBDAI_LIBRARY})
   target_link_libraries(_inference ${GMPXX_LIBRARY})
   target_link_libraries(_inference ${GMP_LIBRARY}) 
endif()


if(WITH_CPLEX)
  if(WIN32)
      target_link_libraries(_inference wsock32.lib ${ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CONCERT_LIBRARY})
   else()
      target_link_libraries(_inference ${CMAKE_THREAD_LIBS_INIT} ${ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CONCERT_LIBRARY} )
      find_library(RT rt)
    endif()
endif()



IF(PYTHON_PLATFORM MATCHES "^windows$")
        SET_TARGET_PROPERTIES(_inference PROPERTIES OUTPUT_NAME "inference"  PREFIX "" SUFFIX  ".pyd")
ELSEIF(MACOSX)
    SET_TARGET_PROPERTIES(_inference PROPERTIES OUTPUT_NAME "inference" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_inference PROPERTIES OUTPUT_NAME "inference"   PREFIX "_")
ENDIF()






#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy __init__.py" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()
