IF(BUILD_COMMANDLINE)
  add_subdirectory(commandline)
ENDIF()
if(BUILD_PYTHON_WRAPPER)
   add_subdirectory(python)
endif() 
IF(BUILD_MATLAB_WRAPPER)
  add_subdirectory(matlab)
ENDIF()


