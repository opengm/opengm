
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
add_library(_opengmcore SHARED  
opengmcore.cpp
pyConfig.cxx
pyGm.cxx
pyFactor.cxx
pyIfactor.cxx
pyFid.cxx
pyFunctionTypes.cxx
pySpace.cxx
pyVector.cxx
pyEnum.cxx
)





#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries(_opengmcore ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARIES} rt)
IF(WITH_MAXFLOW)
   target_link_libraries(_opengmcore  external-library-maxflow)
endif(WITH_MAXFLOW)
if(WITH_HDF5)
   target_link_libraries(_opengmcore  ${HDF5_CORE_LIBRARY} ${HDF5_LIBRARIES} )
endif()


IF(PYTHON_PLATFORM MATCHES "^windows$")
        SET_TARGET_PROPERTIES(_opengmcore PROPERTIES OUTPUT_NAME "opengmcore"  PREFIX "" SUFFIX  ".pyd")
ELSEIF(MACOSX)
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES OUTPUT_NAME "opengmcore" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES OUTPUT_NAME "opengmcore"   PREFIX "_")
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
