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
if ( WITH_BOOST )
    # Note:
    #  'FIND_PACKAGE(PythonLibs)' is unreliable because results are often inconsistent
    #  with the Python interpreter found previously (e.g. libraries or includes
    #  from incompatible installations). Thus, we ask Python itself for the information.
    #

    ######################################################################
    #
    #      find Python prefix
    #
    ######################################################################
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "import sys; print(sys.exec_prefix)"
                      OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "import sys; print(sys.version.split()[0])"
                      OUTPUT_VARIABLE PYTHON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "import sys; print(sys.version_info[0])"
                      OUTPUT_VARIABLE PYTHON_VERSION_MAJOR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "import sys; print(sys.version_info[1])"
                      OUTPUT_VARIABLE PYTHON_VERSION_MINOR OUTPUT_STRIP_TRAILING_WHITESPACE)

    MESSAGE(STATUS "Using Python ${PYTHON_VERSION} at ${PYTHON_EXECUTABLE}")

    ######################################################################
    #
    #      find Python includes
    #
    ######################################################################
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                    "from distutils.sysconfig import *; print(get_python_inc())"
                    OUTPUT_VARIABLE PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
    SET(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE}
        CACHE PATH "Path to Python include files"
        FORCE)

    IF(PYTHON_INCLUDE_DIRS)
        MESSAGE(STATUS "Found Python includes:  ${PYTHON_INCLUDE_DIRS}")
    ELSE()
        MESSAGE(STATUS "Could NOT find Python includes")
    ENDIF()

    ######################################################################
    #
    #      find Python library
    #
    ######################################################################
    IF(APPLE AND ${PYTHON_PREFIX} MATCHES ".*framework.*")
        SET(PYTHON_LIBRARIES "${PYTHON_PREFIX}/Python"
            CACHE FILEPATH "Python libraries"
            FORCE)
    ELSE()
        IF(WIN32)
            set(PYTHON_LIBRARY_NAME python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
        ELSE()
            execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                             "from distutils.sysconfig import *; print(get_config_var('LDLIBRARY'))"
                              OUTPUT_VARIABLE PYTHON_LIBRARY_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                             "from distutils.sysconfig import *; print(get_config_var('LIBDIR'))"
                               OUTPUT_VARIABLE PYTHON_LIBRARY_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        ENDIF()
        FIND_LIBRARY(PYTHON_LIBRARIES ${PYTHON_LIBRARY_NAME} HINTS "${PYTHON_LIBRARY_PREFIX}" "${PYTHON_PREFIX}"
                     PATH_SUFFIXES lib lib64 libs DOC "Python libraries")
        unset(PYTHON_LIBRARY_PREFIX)
    ENDIF()

    IF(PYTHON_LIBRARIES)
        MESSAGE(STATUS "Found Python library: ${PYTHON_LIBRARIES}")
    ELSE()
        MESSAGE(STATUS "Could NOT find Python library")
    ENDIF()

    ######################################################################
    #
    #      find boost::python library
    #
    ######################################################################
    # 'FIND_PACKAGE(Boost COMPONENTS python)' is unreliable because it often selects
    # boost_python for the wrong Python version
    IF(Boost_FOUND)
        IF(Boost_USE_MULTITHREADED)
            # define names for thread-safe library variants
            SET(BOOST_PYTHON_NAMES
                    boost_python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-mt
                    boost_python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-mt
                    boost_python${PYTHON_VERSION_MAJOR}-mt
                    boost_python-mt)
        ENDIF()
        IF(Boost_LIB_SUFFIX)
            SET(BOOST_PYTHON_NAMES ${BOOST_PYTHON_NAMES}
                # Windows with mangled library names
                boost_python${PYTHON_VERSION_MAJOR}${Boost_LIB_SUFFIX}
                boost_python${Boost_LIB_SUFFIX})
        ENDIF()

        # define names for boost_python library variants
        # (may or may not be thread-safe)
        SET(BOOST_PYTHON_NAMES ${BOOST_PYTHON_NAMES}
                # Linux with multiple Python versions
                boost_python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
                # Gentoo
                boost_python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
                # Mac with Python 3
                boost_python${PYTHON_VERSION_MAJOR}
                # default
                boost_python)

        FIND_LIBRARY(Boost_PYTHON_LIBRARIES
                     NAMES ${BOOST_PYTHON_NAMES}
                     NAMES_PER_DIR
                     HINTS "${Boost_LIBRARY_DIR}"
                     DOC "boost_python libraries")
    ENDIF()

    if(Boost_PYTHON_LIBRARIES)
        MESSAGE(STATUS "Found boost_python library: ${Boost_PYTHON_LIBRARIES}")
    else()
        MESSAGE(STATUS "Could NOT find boost_python library")
    endif()
else()
   message( FATAL_ERROR "Option  WITH_BOOST  needs to be activated to compile the OpenGM Python Wrapper" )
endif()



find_package(NUMPY REQUIRED)

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
add_subdirectory(benchmark)

#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python-opengm files  from src to build" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_misc.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__version__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_visu.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_inf_param.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_to_native_converter.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_inference_parameter_injector.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_inference_injector.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_inference_interface_generator.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/functionhelper.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()


