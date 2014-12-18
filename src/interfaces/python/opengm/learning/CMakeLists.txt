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
set(PY_OPENGM_CORE_SOURCES
            learning.cxx
            pyWeights.cxx
            pyDataset.cxx
            pyLoss.cxx
            pyGridSearchLearner.cxx
            pyStructMaxMarginLearner.cxx)

if(APPLE)
    add_library(_learning MODULE ${PY_OPENGM_CORE_SOURCES})
else()
    add_library(_learning SHARED ${PY_OPENGM_CORE_SOURCES})
endif(APPLE)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
if(MSVC AND NOT(MSVC_VERSION LESS 1400))
    SET_TARGET_PROPERTIES(_learning PROPERTIES COMPILE_FLAGS "/bigobj")
endif()
if(APPLE)
    SET_TARGET_PROPERTIES(_learning PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)


if(LINK_RT)
    find_library(RT rt)
    target_link_libraries(_learning ${Boost_PYTHON_LIBRARIES} rt)
else()
    target_link_libraries(_learning ${Boost_PYTHON_LIBRARIES})
endif(LINK_RT)

set_target_properties(_learning PROPERTIES PREFIX "")



IF(WIN32)
    SET_TARGET_PROPERTIES(_learning PROPERTIES OUTPUT_NAME "learning"  PREFIX "_" SUFFIX  ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES(_learning PROPERTIES OUTPUT_NAME "learning" PREFIX "_" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_learning PROPERTIES OUTPUT_NAME "learning"   PREFIX "_")
ENDIF()

target_link_libraries(_learning  ${HDF5_CORE_LIBRARY} ${HDF5_LIBRARIES} )




#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python-learning files  from src to build" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()
