add_subdirectory(opengm)
if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy test.py" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()


SET(OPENGM_PYTHON_BUILD_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR})
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source/conf.py.in ${CMAKE_CURRENT_SOURCE_DIR}/docsrc/source/conf.py  @ONLY)

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python docsrc" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/docsrc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()




