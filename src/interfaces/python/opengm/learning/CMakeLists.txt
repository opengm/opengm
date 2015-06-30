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
set(PY_OPENGM_CORE_SOURCES
            learning.cxx
            pyLFunctionGen.cxx
            pyWeights.cxx
            pyDataset.cxx
            pyLoss.cxx
            pyGridSearchLearner.cxx
            pyMaxLikelihoodLearner.cxx
            pyStructMaxMarginLearner.cxx
            pySubgradientSSVM.cxx
            pyStructPerceptron.cxx
            pyRws.cxx
            )

if(APPLE)
    add_library(_learning MODULE ${PY_OPENGM_CORE_SOURCES})
else()
    add_library(_learning SHARED ${PY_OPENGM_CORE_SOURCES})
endif(APPLE)


#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
if(OPENMP_FOUND)
    SET_TARGET_PROPERTIES(_learning PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
    SET_TARGET_PROPERTIES(_learning PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

if(MSVC AND NOT(MSVC_VERSION LESS 1400))
    SET_PROPERTY(TARGET _learning APPEND_STRING PROPERTY COMPILE_FLAGS " /bigobj")
endif()
if(APPLE)
    SET_PROPERTY(TARGET _learning APPEND_STRING PROPERTY LINK_FLAGS " -undefined dynamic_lookup")
endif(APPLE)


if(LINK_RT)
    find_library(RT rt)
    target_link_libraries(_learning ${Boost_PYTHON_LIBRARIES} rt)
else()
    target_link_libraries(_learning ${Boost_PYTHON_LIBRARIES})
endif(LINK_RT)

set_target_properties(_learning PROPERTIES PREFIX "")


IF(WIN32)
    SET_TARGET_PROPERTIES(_learning PROPERTIES OUTPUT_NAME "learning"  PREFIX "_" SUFFIX  ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES(_learning PROPERTIES OUTPUT_NAME "learning" PREFIX "_" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES(_learning PROPERTIES OUTPUT_NAME "learning"   PREFIX "_")
ENDIF()

target_link_libraries(_learning  ${HDF5_CORE_LIBRARY} ${HDF5_LIBRARIES} )



if(WITH_LIBDAI)
  target_link_libraries(_learning ${LIBDAI_LIBRARY})
  target_link_libraries(_learning ${GMPXX_LIBRARY})
  target_link_libraries(_learning ${GMP_LIBRARY}) 
endif()

if(WITH_QPBO)
   target_link_libraries(_learning external-library-qpbo-shared)
endif()

if(WITH_MAXFLOW)
   target_link_libraries(_learning external-library-maxflow-shared)
endif()

if(WITH_MAXFLOW_IBFS)
  target_link_libraries(_learning external-library-maxflow-ibfs-shared)
endif()

if(WITH_TRWS)
   target_link_libraries(_learning external-library-trws-shared)
endif()


if(WITH_FASTPD)
   target_link_libraries(_learning external-library-fastpd-shared)
endif()

if(WITH_AD3)
   target_link_libraries(_learning external-library-ad3-shared )
endif()

#SET(LINK_FLAGS "${LINK_FLAGS} -PIC")
#SET_TARGET_PROPERTIES(_learning PROPERTIES LINK_FLAGS   "-fPIC")
#add_definitions(-fPIC)

if(WITH_CONICBUNDLE)
  #target_link_libraries(_learning ${CONICBUNDLE_LIBRARY})
endif()

if(WITH_MRF)
   target_link_libraries(_learning external-library-mrf-shared)
endif()




if(WITH_CPLEX)
  if(WIN32)
      target_link_libraries(_learning wsock32.lib ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CPLEX_CONCERT_LIBRARY})
   else()
      target_link_libraries(_learning ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CPLEX_CONCERT_LIBRARY} )
    endif()
endif()


if(WITH_GUROBI)
  target_link_libraries(_learning ${CMAKE_THREAD_LIBS_INIT} 
    ${GUROBI_LIBRARIES}
    #${GUOBI_CXX_LIBRARY}  
    ${CMAKE_THREAD_LIBS_INIT}
  )
endif()

#--------------------------------------------------------------
# Copy from src to build
#--------------------------------------------------------------

if( ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL  ${CMAKE_CURRENT_BINARY_DIR} )
   message(STATUS "same src and build dir.")
else()
   message(STATUS "copy python-learning files  from src to build" )
   file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
endif()
