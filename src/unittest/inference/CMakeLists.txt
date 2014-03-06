ADD_EXECUTABLE(test-adsal test_adsal.cxx ${headers})
add_test(test-adsal ${CMAKE_CURRENT_BINARY_DIR}/test-adsal)
 
 ADD_EXECUTABLE(test-nesterov test_nesterov.cxx ${headers})
 add_test(test-nesterov ${CMAKE_CURRENT_BINARY_DIR}/test-nesterov) 
#  if(WITH_HDF5)
#      add_executable(test-nesterov test_nesterov.cxx ${headers})
#      target_link_libraries(test-nesterov ${HDF5_LIBRARIES})
#      add_test(test-nesterov ${CMAKE_CURRENT_BINARY_DIR}/test-nesterov)
#   endif()
  
ADD_EXECUTABLE(test-trwsi test_trwsi.cxx ${headers})
add_test(test-trwsi ${CMAKE_CURRENT_BINARY_DIR}/test-trwsi)

add_executable(test-messagepassing test_messagepassing.cxx ${headers})
add_test(test-messagepassing ${CMAKE_CURRENT_BINARY_DIR}/test-messagepassing)

add_executable(test-dynamicprogramming test_dynamicprogramming.cxx ${headers})
add_test(test-dynamicprogramming ${CMAKE_CURRENT_BINARY_DIR}/test-dynamicprogramming)

#add_executable(test-gibbs test_gibbs.cxx ${headers})
#add_test(test-gibbs ${CMAKE_CURRENT_BINARY_DIR}/test-gibbs)

#add_executable(test-swendsenwang test_swendsenwang.cxx ${headers})
#add_test(test-swendsenwang ${CMAKE_CURRENT_BINARY_DIR}/test-swendsenwang)

#add_executable(test-pbp test_pbp.cxx ${headers})
#add_test(test-pbp ${CMAKE_CURRENT_BINARY_DIR}/test-pbp)



if(WITH_AD3) 
  add_executable(test-loc test_loc.cxx ${headers})
  target_link_libraries(test-loc external-library-ad3 )
  if(LINK_RT)
   find_library(RT rt)
   target_link_libraries(test-loc rt)
  endif(LINK_RT)

  if(WITH_CPLEX)
  if(WIN32)
    target_link_libraries(test-loc wsock32.lib ${CPLEX_LIBRARIES} )
  else()
    target_link_libraries(test-loc ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_LIBRARIES} )
  endif()
endif()



  add_test(test-loc ${CMAKE_CURRENT_BINARY_DIR}/test-loc)
endif()



add_executable(test-icm test_icm.cxx ${headers})
add_test(test-icm ${CMAKE_CURRENT_BINARY_DIR}/test-icm)

add_executable(test-bruteforce test_bruteforce.cxx ${headers})
add_test(test-bruteforce ${CMAKE_CURRENT_BINARY_DIR}/test-bruteforce)

add_executable(test-lazyflipper test_lazyflipper.cxx ${headers})
add_test(test-lazyflipper  ${CMAKE_CURRENT_BINARY_DIR}/test-lazyflipper)

add_executable(test-movemaker test_movemaker.cxx ${headers})
if(LINK_RT)
   find_library(RT rt)
   target_link_libraries(test-movemaker rt)
endif(LINK_RT)
add_test(test-movemaker ${CMAKE_CURRENT_BINARY_DIR}/test-movemaker)


add_executable(test-dualdecomposition test_dualdecomposition.cxx ${headers})
add_test(test-dualdecomposition ${CMAKE_CURRENT_BINARY_DIR}/test-dualdecomposition)
if(LINK_RT)
  find_library(RT rt)
  target_link_libraries(test-dualdecomposition rt)
  if(WITH_CONICBUNDLE)
    target_link_libraries(test-dualdecomposition ${CONICBUNDLE_LIBRARY})
  endif()
endif(LINK_RT)
if(APPLE AND WITH_CONICBUNDLE)
  target_link_libraries(test-dualdecomposition ${CONICBUNDLE_LIBRARY})
endif()

add_executable(test-astar test_astar.cxx ${headers})
if(LINK_RT)
   find_library(RT rt)
   target_link_libraries(test-astar rt)
endif(LINK_RT)
add_test(test-astar  ${CMAKE_CURRENT_BINARY_DIR}/test-astar)

if(WITH_BOOST)
   add_executable(test-2sat test_2sat.cxx ${headers})
   add_test(test-2sat ${CMAKE_CURRENT_BINARY_DIR}/test-2sat)
endif()

if(WITH_BOOST OR WITH_MAXFLOW OR WITH_MAXFLOW_IBFS)
   add_executable(test-minstcut test_minstcut.cxx ${headers})
   add_executable(test-graphcut test_graphcut.cxx ${headers})
   add_executable(test-alphaexpansion test_alphaexpansion.cxx ${headers})
   add_executable(test-alphabetaswap test_alphabetaswap.cxx ${headers})
   add_executable(test-qpbo test_qpbo.cxx ${headers})
   IF(WITH_MAXFLOW)
      target_link_libraries(test-minstcut external-library-maxflow)
      target_link_libraries(test-graphcut external-library-maxflow)
      target_link_libraries(test-alphaexpansion external-library-maxflow)
      target_link_libraries(test-alphabetaswap external-library-maxflow)
      target_link_libraries(test-qpbo external-library-maxflow)
   endif(WITH_MAXFLOW)
   IF(WITH_MAXFLOW_IBFS)
     target_link_libraries(test-graphcut external-library-maxflow-ibfs)
   endif(WITH_MAXFLOW_IBFS)
   add_test(test-minstcut  ${CMAKE_CURRENT_BINARY_DIR}/test-minstcut)
   add_test(test-graphcut  ${CMAKE_CURRENT_BINARY_DIR}/test-graphcut)
   add_test(test-alphabetaswap  ${CMAKE_CURRENT_BINARY_DIR}/test-alphabetaswap)
   add_test(test-alphaexpansion  ${CMAKE_CURRENT_BINARY_DIR}/test-alphaexpansion)
   add_test(test-qpbo ${CMAKE_CURRENT_BINARY_DIR}/test-qpbo)
endif()

if(WITH_CPLEX)
  add_executable(test-lpcplex test_lpcplex.cxx ${headers})
  #add_executable(test-lpcplex2 test_lpcplex2.cxx ${headers})
  add_executable(test-combilp test_combilp.cxx ${headers})
  if(WIN32)
    target_link_libraries(test-lpcplex wsock32.lib ${CPLEX_LIBRARIES} )
    #target_link_libraries(test-lpcplex2 wsock32.lib ${CPLEX_LIBRARIES} )
    target_link_libraries(test-combilp wsock32.lib ${CPLEX_LIBRARIES} ${HDF5_LIBRARIES} ) 
  else()
    target_link_libraries(test-lpcplex ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_LIBRARIES} )
    #target_link_libraries(test-lpcplex2 ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_LIBRARIES}
    target_link_libraries(test-combilp ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_LIBRARIES} ${HDF5_LIBRARIES}) 
  endif()
  if(LINK_RT)
    target_link_libraries(test-combilp rt)
  endif(LINK_RT)
  
  add_test(test-lpcplex ${CMAKE_CURRENT_BINARY_DIR}/test-lpcplex)
  #add_test(test-lpcplex2 ${CMAKE_CURRENT_BINARY_DIR}/test-lpcplex2)
  add_test(test-combilp ${CMAKE_CURRENT_BINARY_DIR}/test-combilp)
endif()


if(WITH_GUROBI)
  set (CMAKE_CXX_FLAGS "-fno-stack-protector")
  set (CMAKE_LINKER_FLAGS "-lm")
  set (CMAKE_LINK_FLAGS "-lm")

  #add_executable(test-lpgurobi test_lpgurobi.cxx ${headers})
  #target_link_libraries(test-lpgurobi ${CMAKE_THREAD_LIBS_INIT} 
  #  ${GUROBI_LIBRARIES}
  #  #${GUOBI_CXX_LIBRARY}  
  #  #${GUROBI_LIBRARY}
  #  ${CMAKE_THREAD_LIBS_INIT}
  #  #${GUROBI_LIBRARIES}
  #  #
  #)
  #add_test(test-lpgurobi ${CMAKE_CURRENT_BINARY_DIR}/test-lpgurobi)

  message(STATUS "GRB LB: ${GUROBI_LIBRARIES}")
endif()



if(WITH_CPLEX)
  if(WITH_BOOST OR CMAKE_COMPILER_IS_GNUCC)
    add_executable(test-multicut test_multicut.cxx ${headers})
    if(WIN32)
      target_link_libraries(test-multicut wsock32.lib ${CPLEX_LIBRARIES} )  
    else()
      target_link_libraries(test-multicut ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_LIBRARIES} )
    endif()
    if(LINK_RT)
      target_link_libraries(test-multicut rt)
    endif(LINK_RT) 
    add_test(test-multicut ${CMAKE_CURRENT_BINARY_DIR}/test-multicut)
  endif()
endif()

if(WITH_QPBO) 
   add_executable(test-qpbo-external test_qpbo_external.cxx ${headers})
   target_link_libraries(test-qpbo-external external-library-qpbo)
   add_test(test-qpbo-external ${CMAKE_CURRENT_BINARY_DIR}/test-qpbo-external)

   add_executable(test-mqpbo test_mqpbo.cxx ${headers})
   target_link_libraries(test-mqpbo external-library-qpbo)
   add_test(test-mqpbo ${CMAKE_CURRENT_BINARY_DIR}/test-mqpbo)
endif()

if(WITH_BOOST)
  if(WITH_QPBO)
    add_executable(test-rinf test_rinf.cxx ${headers})
    target_link_libraries(test-rinf external-library-qpbo)
    add_test(test-rinf ${CMAKE_CURRENT_BINARY_DIR}/test-rinf)
  endif()
endif()

if(WITH_AD3) 
   add_executable(test-ad3-external test_ad3_external.cxx ${headers})
   target_link_libraries(test-ad3-external external-library-ad3 )
   add_test(test-ad3-external ${CMAKE_CURRENT_BINARY_DIR}/test-ad3-external)
endif()


if(WITH_BOOST)
  if(WITH_QPBO)
    add_executable(test-alphaexpansionfusion test_alphaexpansionfusion.cxx ${headers})
    target_link_libraries(test-alphaexpansionfusion external-library-qpbo)
    add_test(test-alphaexpansionfusion ${CMAKE_CURRENT_BINARY_DIR}/test-alphaexpansionfusion)
  endif()
endif()

if(WITH_TRWS)
   add_executable(test-trws test_trws.cxx ${headers})
   target_link_libraries(test-trws external-library-trws)
   add_test(test-trws ${CMAKE_CURRENT_BINARY_DIR}/test-trws)
endif()


if(WITH_MRF)
   add_executable(test-mrflib test_mrflib.cxx ${headers})
   target_link_libraries(test-mrflib external-library-mrf)
   add_test(test-mrflib ${CMAKE_CURRENT_BINARY_DIR}/test-mrflib)
endif()

if(WITH_LIBDAI)
   add_executable(test-opengm-libdai-interface test_external_libdai.cxx ${headers})
   target_link_libraries(test-opengm-libdai-interface ${LIBDAI_LIBRARY})
   target_link_libraries(test-opengm-libdai-interface ${GMPXX_LIBRARY})
   target_link_libraries(test-opengm-libdai-interface ${GMP_LIBRARY}) 
   add_test(test-opengm-libdai-interface  ${CMAKE_CURRENT_BINARY_DIR}/test-opengm-libdai-interface)
endif()

if(WITH_GCO)
   add_executable(test-gco test_gco.cxx ${headers})
   target_link_libraries(test-gco external-library-gco)
   add_test(test-gco ${CMAKE_CURRENT_BINARY_DIR}/test-gco)
endif()

if(WITH_MPLP)
   add_executable(test-mplp-external test_mplp_external.cxx ${headers})
   target_link_libraries(test-mplp-external external-library-mplp)
   add_test(test-mplp-external ${CMAKE_CURRENT_BINARY_DIR}/test-mplp-external)
endif()
