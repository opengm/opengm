if(WITH_MATLAB)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMATLAB_MEX_FILE -fPIC -ansi -shared -Wl,--version-script,${MATLAB_LIBRARY_DIR}/mexFunction.map")  

  add_executable(opengmBuild.${MATLAB_MEXEXT} opengmBuild.cxx ${headers})
  target_link_libraries(opengmBuild.${MATLAB_MEXEXT} ${HDF5_LIBRARIES}) 

endif(WITH_MATLAB)
