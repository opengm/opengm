 add_executable(buildModel            buildModel.cxx            ${headers})
 add_executable(buildGridModel        buildGridModel.cxx        ${headers})
 add_executable(buildHigherOrderModel buildHigherOrderModel.cxx ${headers})
 add_executable(buildModelEfficent    buildModelEfficent.cxx    ${headers}) 
 add_executable(doMinSumInference     doMinSumInference.cxx     ${headers})
 add_executable(doMaxSumInference     doMaxSumInference.cxx     ${headers})  
 add_executable(doOrAndInference      doOrAndInference.cxx      ${headers}) 
 add_executable(doSumProdInference    doSumProdInference.cxx    ${headers})


if(WITH_TRWS)
  add_executable(doMinSumInferenceExternal     doMinSumInferenceExternal.cxx     ${headers}) 
  target_link_libraries(doMinSumInferenceExternal external-library-trws)
endif() 

if(LINK_RT)
   find_library(RT rt)
   target_link_libraries(buildModelEfficent rt)
 endif(LINK_RT) 

if(WITH_HDF5)
  add_executable(modelIO    modelIO.cxx    ${headers}) 
  target_link_libraries(modelIO ${HDF5_LIBRARIES})
endif(WITH_HDF5)
