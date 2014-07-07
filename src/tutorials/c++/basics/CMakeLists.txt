 add_executable(buildModel            buildModel.cxx            ${headers})
 add_executable(buildGridModel        buildGridModel.cxx        ${headers})
 add_executable(buildHigherOrderModel buildHigherOrderModel.cxx ${headers})
 add_executable(buildModelEfficent    buildModelEfficent.cxx    ${headers}) 
 
if(LINK_RT)
   find_library(RT rt)
   target_link_libraries(buildModelEfficent rt)
 endif(LINK_RT) 

if(WITH_HDF5)
  add_executable(modelIO    modelIO.cxx    ${headers}) 
  target_link_libraries(modelIO ${HDF5_LIBRARIES})
endif(WITH_HDF5)
