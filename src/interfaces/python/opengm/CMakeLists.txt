if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # a lot of warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
elseif(MSVC)
   # how to stripe wihtn msvc
   #add_definitions(-Ds) ????
endif()



#--------------------------------------------------------------
# Find packages
#--------------------------------------------------------------
find_package(Boost COMPONENTS python REQUIRED)
find_package (PythonLibs REQUIRED)
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






add_subdirectory(opengmcore)
add_subdirectory(inference)
add_subdirectory(hdf5)


#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy __init__.py" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__version__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()


