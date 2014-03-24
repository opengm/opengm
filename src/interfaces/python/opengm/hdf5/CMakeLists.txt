#--------------------------------------------------------------
# Find packages
#--------------------------------------------------------------


 message(STATUS "build opengm python with HDF5 support")
 if ( WITH_HDF5 )
   #find_package(HDF5 REQUIRED)
 else()
   message( FATAL_ERROR "Option  WITH_HDF5  needs to be activated to compile the OpenGM Python Wrapper" )
 endif()
 find_package(NUMPY) 
 include_directories(${HDF5_INCLUDE_DIR})
 add_definitions(${HDF5_CPPFLAGS})
 add_definitions(-DWITH_HDF5)

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
if(APPLE)
    add_library(_hdf5 MODULE 
    hdf5.cpp
    pyHdf5.cxx
    )
else()
    add_library(_hdf5 SHARED  
    hdf5.cpp
    pyHdf5.cxx
    )
endif(APPLE)





#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
if(MSVC AND NOT(MSVC_VERSION LESS 1400))
    SET_TARGET_PROPERTIES(_hdf5 PROPERTIES COMPILE_FLAGS "/bigobj")
endif()
if(APPLE)
    SET_TARGET_PROPERTIES(_hdf5 PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)
target_link_libraries(_hdf5 ${Boost_PYTHON_LIBRARIES}  ${HDF5_LIBRARIES})
set_target_properties(_hdf5 PROPERTIES PREFIX "")



IF(WIN32)
    SET_TARGET_PROPERTIES(_hdf5 PROPERTIES OUTPUT_NAME "hdf5"  PREFIX "_" SUFFIX  ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES(_hdf5 PROPERTIES OUTPUT_NAME "hdf5" PREFIX "_" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_hdf5 PROPERTIES OUTPUT_NAME "hdf5"   PREFIX "_")
ENDIF()



#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python-hdf5 files  from src to build" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()


