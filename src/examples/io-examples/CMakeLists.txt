if(WITH_HDF5 AND HDF5_FOUND)
   add_executable(example-io io_graphical_model.cxx ${headers})
   if(WIN32 OR APPLE)
      target_link_libraries(example-io ${HDF5_LIBRARIES})
   else()
      target_link_libraries(example-io ${HDF5_LIBRARIES} rt)
   endif()
endif()


