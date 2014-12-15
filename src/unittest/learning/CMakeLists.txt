
add_definitions(-DOPENGM_DEBUG)

if(BUILD_TESTING)
   if(WITH_HDF5)
      add_executable(test-dataset-io test_dataset_io.cxx ${headers})
      target_link_libraries(test-dataset-io ${HDF5_LIBRARIES})
      add_test(test-dataset-io ${CMAKE_CURRENT_BINARY_DIR}/test-dataset-io)

      add_executable(test-dataset test_dataset.cxx ${headers})
      target_link_libraries(test-dataset ${HDF5_LIBRARIES})
      add_test(test-dataset ${CMAKE_CURRENT_BINARY_DIR}/test-dataset)
   endif()
  
endif()
