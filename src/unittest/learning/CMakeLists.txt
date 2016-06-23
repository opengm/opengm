
add_definitions(-DOPENGM_DEBUG)

if(BUILD_TESTING)

    add_executable(test-gridsearch-learner test_gridsearch_learner.cxx ${headers})
    add_test(test-gridsearch-learner ${CMAKE_CURRENT_BINARY_DIR}/test-gridsearch-learner)

    add_executable(test-maximum-likelihood-learner test_maximum_likelihood_learner.cxx ${headers})
    add_test(test-maximum-likelihood-learner ${CMAKE_CURRENT_BINARY_DIR}/test-maximum-likelihood-learner)

    if(WITH_HDF5)
        add_executable(test-dataset-io test_dataset_io.cxx ${headers})
        target_link_libraries(test-dataset-io ${HDF5_LIBRARIES})
        add_test(test-dataset-io ${CMAKE_CURRENT_BINARY_DIR}/test-dataset-io)

        add_executable(test-dataset test_dataset.cxx ${headers})
        target_link_libraries(test-dataset ${HDF5_LIBRARIES})
        add_test(test-dataset ${CMAKE_CURRENT_BINARY_DIR}/test-dataset)
    endif()

    if(WITH_TRWS)
        if(WITH_GUROBI)
            ADD_EXECUTABLE(test-learning test_learning.cxx ${headers})
            target_link_libraries(test-learning ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
            target_link_libraries(test-learning external-library-trws)
            add_test(test-learning ${CMAKE_CURRENT_BINARY_DIR}/test-learning) 
        else()
            if(WITH_CPLEX)
                ADD_EXECUTABLE(test-learning test_learning.cxx ${headers})
                target_link_libraries(test-learning ${CPLEX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
                target_link_libraries(test-learning external-library-trws)
                add_test(test-learning ${CMAKE_CURRENT_BINARY_DIR}/test-learning)
            endif()
        endif()
    endif()


    if(WITH_CPLEX)
        ADD_EXECUTABLE(test-subgradient-ssvm test_subgradient_ssvm.cxx ${headers})
        if(OPENMP_FOUND)
            SET_TARGET_PROPERTIES(test-subgradient-ssvm PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
            SET_TARGET_PROPERTIES(test-subgradient-ssvm PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
        endif()
        target_link_libraries(test-subgradient-ssvm ${CPLEX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
        add_test(test-subgradient-ssvm ${CMAKE_CURRENT_BINARY_DIR}/test-subgradient-ssvm)
    endif()

  add_executable(test-generalized-hammingloss test_generalized_hammingloss.cxx ${headers})
  add_test(test-generalized-hammingloss ${CMAKE_CURRENT_BINARY_DIR}/test-generalized-hammingloss)
endif()
