#--------------------------------------------------------------
# Include Directories
#--------------------------------------------------------------
find_package(NUMPY)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
	 ${PYTHON_INCLUDE_DIRS}
	 ${PYTHON_NUMPY_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${Boost_PYTHON_INCLUDE_DIR}
)





#--------------------------------------------------------------
# Add opengmcore library
#--------------------------------------------------------------
if(APPLE)
    add_library(_inference MODULE 
    inference.cpp
    pyInference.cxx
    #pyPbp.cxx
    pyIcm.cxx
    pyGraphcut.cxx
    pyBp.cxx
    pyTrbp.cxx
    pyLoc.cxx
    pyAstar.cxx
    pyDualDecompositionSubGradient.cxx
    pyReducedInference.cxx
#pyDualDecompositionBundle.cxx
#pyDualDecompositionMerit.cxx
#pyGibbs.cxx
    pyBruteforce.cxx
    pyLazyflipper.cxx
    pyAeFusion.cxx
    pyAe.cxx
    pyAbSwap.cxx
    pyDynp.cxx
    pyCplex.cxx
#pyLpInference.cxx
    pyTrws.cxx
    pyQpbo.cxx
    pyMQpbo.cxx
    pyLibdai.cxx
    pyMrf.cxx
    pyFastPD.cxx
    pyAd3.cxx
    pyPartitionMove.cxx
    pyMultiCut.cxx
    #pySelfFusion.cxx
    pyFusionMoves.cxx
     )
else()
    add_library(_inference SHARED  
    inference.cpp
    pyInference.cxx
    #pyPbp.cxx
    pyIcm.cxx
    pyGraphcut.cxx
    pyBp.cxx
    pyTrbp.cxx
    pyLoc.cxx
    pyAstar.cxx
    pyDualDecompositionSubGradient.cxx
    pyReducedInference.cxx
#pyDualDecompositionBundle.cxx
#pyDualDecompositionMerit.cxx
#pyGibbs.cxx
    pyBruteforce.cxx
    pyLazyflipper.cxx
    pyAeFusion.cxx
    pyAe.cxx
    pyAbSwap.cxx
    pyDynp.cxx
    pyCplex.cxx
#pyLpInference.cxx
    pyTrws.cxx
    pyQpbo.cxx
    pyMQpbo.cxx
    pyLibdai.cxx
    pyMrf.cxx
    pyFastPD.cxx
    pyAd3.cxx
    pyPartitionMove.cxx
    pyMultiCut.cxx
    #pySelfFusion.cxx
    pyFusionMoves.cxx
    )
endif(APPLE)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
if(MSVC AND NOT(MSVC_VERSION LESS 1400))
    SET_TARGET_PROPERTIES(_inference PROPERTIES COMPILE_FLAGS "/bigobj")
endif()
if(APPLE)
    SET_TARGET_PROPERTIES(_inference PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)


if(LINK_RT)
    find_library(RT rt)
    target_link_libraries(_inference ${Boost_PYTHON_LIBRARIES} rt)
else()
    target_link_libraries(_inference ${Boost_PYTHON_LIBRARIES})
endif(LINK_RT)

set_target_properties(_inference PROPERTIES PREFIX "")

if(WITH_LIBDAI)
  target_link_libraries(_inference ${LIBDAI_LIBRARY})
  target_link_libraries(_inference ${GMPXX_LIBRARY})
  target_link_libraries(_inference ${GMP_LIBRARY}) 
endif()

if(WITH_QPBO)
   target_link_libraries(_inference external-library-qpbo-shared)
endif()

if(WITH_MAXFLOW)
   target_link_libraries(_inference external-library-maxflow-shared)
endif()

if(WITH_MAXFLOW_IBFS)
  target_link_libraries(_inference external-library-maxflow-ibfs-shared)
endif()

if(WITH_TRWS)
   target_link_libraries(_inference external-library-trws-shared)
endif()


if(WITH_FASTPD)
   target_link_libraries(_inference external-library-fastpd-shared)
endif()

if(WITH_AD3)
   target_link_libraries(_inference external-library-ad3-shared )
endif()

#SET(LINK_FLAGS "${LINK_FLAGS} -PIC")
#SET_TARGET_PROPERTIES(_inference PROPERTIES LINK_FLAGS   "-fPIC")
#add_definitions(-fPIC)

if(WITH_CONICBUNDLE)
  #target_link_libraries(_inference ${CONICBUNDLE_LIBRARY})
endif()

if(WITH_MRF)
   target_link_libraries(_inference external-library-mrf-shared)
endif()

if(WITH_CPLEX)
  if(WIN32)
      target_link_libraries(_inference wsock32.lib ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CPLEX_CONCERT_LIBRARY})
   else()
      target_link_libraries(_inference ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CPLEX_CONCERT_LIBRARY} )
    endif()
endif()



if(WITH_GUROBI)
  target_link_libraries(_inference ${CMAKE_THREAD_LIBS_INIT} 
    ${GUROBI_LIBRARIES}
    #${GUOBI_CXX_LIBRARY}  
    #${GUROBI_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    #${GUROBI_LIBRARIES}
    #
  )
endif()

IF(WIN32)
    SET_TARGET_PROPERTIES(_inference PROPERTIES OUTPUT_NAME "inference"  PREFIX "_" SUFFIX  ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES(_inference PROPERTIES OUTPUT_NAME "inference" PREFIX "_" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_inference PROPERTIES OUTPUT_NAME "inference"   PREFIX "_")
ENDIF()






#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python-inference files  from src to build" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()
