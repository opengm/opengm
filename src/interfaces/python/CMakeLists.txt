#add_definitions(-DNPY_NO_DEPRECATED_API)
#add_definitions(-DNPY_1_7_API_VERSION)

add_subdirectory(opengm)
if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy test.py" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
   cmake_minimum_required(VERSION 2.8)
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/examples )
endif()

SET(OPENGM_PYTHON_BUILD_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR})
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source/conf.py.in ${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source/conf.py  @ONLY)

message(STATUS ${OPENGM_PYTHON_BUILD_MODULE_DIR})

find_package(SPHINX)
if(FALSE AND SPHINX_FOUND)
    message(STATUS "FOUND_SPHINX")

    if(NOT DEFINED SPHINX_THEME)
        set(SPHINX_THEME default)
    endif()
    if(NOT DEFINED SPHINX_THEME_DIR)
        set(SPHINX_THEME_DIR)
    endif()
    # configured documentation tools and intermediate build results
    set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_build)
    # Sphinx cache with pickled ReST documents
    set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/opengm/_doctrees")
    # HTML output directory
    set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/opengm/html")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source/conf.py.in"
        #"${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source/conf.py"
        "${BINARY_BUILD_DIR}/conf.py"
        @ONLY)
    add_custom_target(python-doc ALL
        ${SPHINX_EXECUTABLE}
        -q -b html
        #-c "${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source"
        -c "${BINARY_BUILD_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source"
        "${SPHINX_HTML_DIR}"
        COMMENT "Building HTML documentation with Sphinx")
    add_dependencies(python-doc _opengmcore )
    add_dependencies(python-doc _inference )
    add_dependencies(python-doc _hdf5 )

else()
    message(STATUS "CANNOT Building HTML documentation with Sphinx , did not find Sphinx")
endif()






