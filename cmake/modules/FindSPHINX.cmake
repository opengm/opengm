find_program(SPHINX_EXECUTABLE NAMES sphinx-build
  HINTS
  $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin\r\n  DOC "Sphinx documentation generator"
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx DEFAULT_MSG\r\n  SPHINX_EXECUTABLE
)


IF (SPHINX_EXECUTABLE )
   SET(SPHINX_FOUND TRUE)
ENDIF (SPHINX_EXECUTABLE)

mark_as_advanced(
  SPHINX_EXECUTABLE
)