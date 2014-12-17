#--------------------------------------------------------------
# Find packages
#--------------------------------------------------------------
#find_package(Boost COMPONENTS python REQUIRED)
#find_package (PythonLibs REQUIRED)
if(WITH_HDF5)
#   message(STATUS "build opengm python with HDF5 support")
#   find_package(HDF5 REQUIRED)
#   include_directories(${HDF5_INCLUDE_DIR})
#   add_definitions(${HDF5_CPPFLAGS})
#   add_definitions(-DWITH_HDF5)
#else()
#   message(STATUS "build opengm python without HDF5")
endif()
#--------------------------------------------------------------
# Include Directories
#--------------------------------------------------------------
find_package(NUMPY)
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
if(APPLE)
    add_library(_opengmcore MODULE 
    opengmcore.cpp
    pyGmManipulator.cxx
    pyConfig.cxx
    pyGm.cxx
    pyFactor.cxx
    pyMovemaker.cxx
    pyIfactor.cxx
    pyFid.cxx
    pyFunctionTypes.cxx
    pyFunctionGen.cxx
    pySpace.cxx
    pyVector.cxx
    pyEnum.cxx
    )
else()
    add_library(_opengmcore SHARED  
    opengmcore.cpp
    pyGmManipulator.cxx
    pyConfig.cxx
    pyGm.cxx
    pyFactor.cxx
    pyMovemaker.cxx
    pyIfactor.cxx
    pyFid.cxx
    pyFunctionTypes.cxx
    pyFunctionGen.cxx
    pySpace.cxx
    pyVector.cxx
    pyEnum.cxx
    )
endif(APPLE)




#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
if(MSVC AND NOT(MSVC_VERSION LESS 1400))
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES COMPILE_FLAGS "/bigobj")
endif()
if(APPLE)
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)



if(LINK_RT)
    find_library(RT rt)
    target_link_libraries(_opengmcore ${Boost_PYTHON_LIBRARIES} rt)
else()
    target_link_libraries(_opengmcore ${Boost_PYTHON_LIBRARIES})
endif(LINK_RT)

IF(WITH_MAXFLOW)
   #target_link_libraries(_opengmcore  external-library-maxflow)
endif(WITH_MAXFLOW)
if(WITH_HDF5)
   target_link_libraries(_opengmcore  ${HDF5_CORE_LIBRARY} ${HDF5_LIBRARIES} )
endif()


IF(WIN32)
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES OUTPUT_NAME "opengmcore"  PREFIX "_" SUFFIX  ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES OUTPUT_NAME "opengmcore" PREFIX "_" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_opengmcore PROPERTIES OUTPUT_NAME "opengmcore"   PREFIX "_")
ENDIF()




#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python-opengmcore files  from src to build" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py          DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/factorSubset.py      DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/gm_injector.py       DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/factor_injector.py   DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/function_injector.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/dtypes.py            DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/printing.py          DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shapewalker.py       DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()
