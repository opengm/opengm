
add_definitions(-DOPENGM_DEBUG)

if(BUILD_TESTING)
   add_executable(test-graphicalmodelmanipulator test_graphicalmodelmanipulator.cxx ${headers})
   add_test(test-graphicalmodelmanipulator ${CMAKE_CURRENT_BINARY_DIR}/test-graphicalmodelmanipulator)

   add_executable(test-explicit-storage test_explicit_storage.cxx ${headers})
   add_test(test-explicit-storage ${CMAKE_CURRENT_BINARY_DIR}/test-explicit-storage)

   add_executable(test-types test_types.cxx ${headers})
   add_test(test-types ${CMAKE_CURRENT_BINARY_DIR}/test-types)

   add_executable(test-test-type-sizes-for-hdf5 test_type_sizes_for_hdf5.cxx ${headers})
   add_test(test-test-type-sizes-for-hdf5 ${CMAKE_CURRENT_BINARY_DIR}/test-test-type-sizes-for-hdf5)

   #add_executable(test-sparsemarray test_sparsemarray.cxx ${headers})
   add_test(test-fast-sequence ${CMAKE_CURRENT_BINARY_DIR}/test-fast-sequen)
   add_executable(test-fast-sequence test_fast_sequence.cxx ${headers})
   add_test(test-fast-sequence ${CMAKE_CURRENT_BINARY_DIR}/test-fast-sequence)

   add_executable(test-functions test_functions.cxx ${headers})
   add_test(test-functions ${CMAKE_CURRENT_BINARY_DIR}/test-functions)

   add_executable(test-factor test_factor.cxx ${headers})
   add_test(test-factor ${CMAKE_CURRENT_BINARY_DIR}/test-factor)

   add_executable(test-discretespaces test_discretespaces.cxx ${headers})
   add_test(test-discretespaces ${CMAKE_CURRENT_BINARY_DIR}/test-discretespaces)

   add_executable(test-graphicalmodel test_graphicalmodel.cxx ${headers})
   add_test(test-graphicalmodel ${CMAKE_CURRENT_BINARY_DIR}/test-graphicalmodel)

   add_executable(test-factorgraph test_factorgraph.cxx ${headers})
   add_test(test-factorgraph ${CMAKE_CURRENT_BINARY_DIR}/test-factorgraph)

   add_executable(test-graphicalmodeldecomposer test_graphicalmodeldecomposer.cxx ${headers})
   add_test(test-graphicalmodeldecomposer ${CMAKE_CURRENT_BINARY_DIR}/test-graphicalmodeldecomposer)

   add_executable(test-operate test_operate.cxx ${headers})
   add_test(test-operate ${CMAKE_CURRENT_BINARY_DIR}/test-operate)

   add_executable(test-operations test_operations.cxx ${headers})
   add_test(test-operations ${CMAKE_CURRENT_BINARY_DIR}/test-operations)

   add_executable(test-accumulate test_accumulate.cxx ${headers})
   add_test(test-accumulate ${CMAKE_CURRENT_BINARY_DIR}/test-accumulate)

   add_executable(test-randomaccessset test_randomaccessset.cxx ${headers})
   add_test(test-randomaccessset ${CMAKE_CURRENT_BINARY_DIR}/test-randomaccessset)

   add_executable(test-tribool test_tribool.cxx ${headers})
   add_test(test-tribool ${CMAKE_CURRENT_BINARY_DIR}/test-tribool)

   add_executable(test-accessor-iterator test_accessor_iterator.cxx ${headers})
   add_test(test-accessor-iterator ${CMAKE_CURRENT_BINARY_DIR}/test-accessor-iterator)

   add_executable(test-random test_random_1.cxx ${headers})
   add_test(test-random ${CMAKE_CURRENT_BINARY_DIR}/test-random) 

   ADD_EXECUTABLE(test-primal-lpbound test_primal_lpbound.cpp ${headers}) 
   add_test(test-primal-lpbound ${CMAKE_CURRENT_BINARY_DIR}/test-primal-lpbound) 

   ADD_EXECUTABLE(test-transportsolver test_transportsolver.cpp ${headers})
   add_test(test-transportsolver ${CMAKE_CURRENT_BINARY_DIR}/test-transportsolver) 

   if(WITH_HDF5)
      add_executable(test-io-hdf5 test_io_hdf5.cxx ${headers})
      target_link_libraries(test-io-hdf5 ${HDF5_LIBRARIES})
      add_test(test-io-hdf5 ${CMAKE_CURRENT_BINARY_DIR}/test-io-hdf5)
   endif()

   add_subdirectory(inference)
endif()
