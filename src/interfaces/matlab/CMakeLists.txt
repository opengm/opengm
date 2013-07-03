if(WITH_MATLAB)
  include_directories(${MATLAB_INCLUDE_DIR})
  add_definitions(-DWITH_MATLAB)
endif(WITH_MATLAB)

add_subdirectory(opengm)

