FIND_PATH(CONICBUNDLE_INCLUDE_DIR
  CBSolver.hxx
  HINTS "src/external/ConicBundle-v0.3.11.src-patched/include/" 
)

if(WIN32)
    FIND_LIBRARY(CONICBUNDLE_LIBRARY
       cb.lib   #TODO check lib name for windows
       HINTS "src/external/ConicBundle-v0.3.11.src-patched/lib/" 
    )
else()
    FIND_LIBRARY(CONICBUNDLE_LIBRARY
       libcb.a
       HINTS "src/external/ConicBundle-v0.3.11.src-patched/lib/" 
    )
endif()
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CONICBUNDLE
   DEFAULT_MSG 
   CONICBUNDLE_LIBRARY 
   CONICBUNDLE_INCLUDE_DIR 
)

IF(CONICBUNDLE_FOUND)

ELSE()
   SET(CONICBUNDLE_INCLUDE_DIR CONICBUNDLE_INCLUDE_DIR-NOTFOUND)
   SET(CONICBUNDLE_LIBRARY     CONICBUNDLE_LIBRARY-NOTFOUND)
ENDIF(CONICBUNDLE_FOUND)
