if(WITH_MATLAB)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMATLAB_MEX_FILE -fPIC -ansi -shared -Wl,--version-script,${MATLAB_LIBRARY_DIR}/mexFunction.map")  

  add_executable(opengmBuild.${MATLAB_MEXEXT} opengmBuild.cxx ${headers})
  target_link_libraries(opengmBuild.${MATLAB_MEXEXT} ${HDF5_LIBRARIES}) 

  add_executable(opengmBuildGrid.mexa64 opengmBuildGrid.cxx ${headers})
  target_link_libraries(opengmBuildGrid.mexa64 ${HDF5_LIBRARIES})

  add_executable(opengmBuildFGE.mexa64 opengmBuildFGE.cxx ${headers})
  target_link_libraries(opengmBuildFGE.mexa64 ${HDF5_LIBRARIES})

  add_executable(opengmBuildMatching.mexa64 opengmBuildMatching.cxx ${headers})
  target_link_libraries(opengmBuildMatching.mexa64 ${HDF5_LIBRARIES}) 
 
  add_executable(readCC.mexa64 readCC.cxx ${headers})
  target_link_libraries(readCC.mexa64 ${HDF5_LIBRARIES}) 

endif(WITH_MATLAB)
