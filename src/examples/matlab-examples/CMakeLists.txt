if(WITH_MATLAB)
  set(MATLAB "/server/opt/matlab" CACHE STRING "Matlb path") 
  SET(MEXEXT "mexa64" CACHE STRING "Matlab MEX file extension")
  FIND_PATH(MATLAB_LIBRARY_DIR "mexFunction.map" ${MATLAB}/extern/lib/glnxa64/)  
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMATLAB_MEX_FILE -fPIC -ansi -shared -Wl,--version-script,${MATLAB_LIBRARY_DIR}/mexFunction.map")  

  add_executable(opengmBuild.mexa64 opengmBuild.cxx ${headers})
  target_link_libraries(opengmBuild.mexa64 ${HDF5_LIBRARIES}) 

endif(WITH_MATLAB)
