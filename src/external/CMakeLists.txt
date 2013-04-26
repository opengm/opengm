add_subdirectory(patches)

IF(WITH_QPBO)
    SET(QPBO_SRC_FILES
        ${QPBO_PATCHEDSRCDIR}/QPBO_extra.cpp
        ${QPBO_PATCHEDSRCDIR}/QPBO.cpp
        ${QPBO_PATCHEDSRCDIR}/QPBO_maxflow.cpp
        ${QPBO_PATCHEDSRCDIR}/QPBO_postprocessing.cpp
    )
    
    SET(QPBO_SRC_FILES_EXIST 1)
    FOREACH(f ${QPBO_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(QPBO_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
     
    IF(DEFINED QPBO_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-qpbo
                  ${QPBO_SRC_FILES}
      )
      ADD_LIBRARY(external-library-qpbo-shared SHARED
                  ${QPBO_SRC_FILES}
      )
    ELSE(DEFINED QPBO_SRC_FILES_EXIST)
        MESSAGE ("QPBO not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED QPBO_SRC_FILES_EXIST)
ENDIF()

IF(WITH_TRWS)
    SET(TRWS_SRC_FILES
        ${TRWS_PATCHEDSRCDIR}/minimize.cpp
        ${TRWS_PATCHEDSRCDIR}/MRFEnergy.cpp
        ${TRWS_PATCHEDSRCDIR}/ordering.cpp
        ${TRWS_PATCHEDSRCDIR}/treeProbabilities.cpp
    )
    
    SET(TRWS_SRC_FILES_EXIST 1)
    FOREACH(f ${TRWS_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(TRWS_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
     
    IF(DEFINED TRWS_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-trws
                  ${TRWS_SRC_FILES}
      )
      ADD_LIBRARY(external-library-trws-shared SHARED
                  ${TRWS_SRC_FILES}
      )
    ELSE(DEFINED TRWS_SRC_FILES_EXIST)
        MESSAGE ("TRWS not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED TRWS_SRC_FILES_EXIST)
ENDIF()

IF(WITH_MAXFLOW)
    SET(MAXFLOW_SRC_FILES
        ${MAXFLOW_PATCHEDSRCDIR}/graph.cpp
        ${MAXFLOW_PATCHEDSRCDIR}/maxflow.cpp
    )
    
    SET(MAXFLOW_SRC_FILES_EXIST 1)
    FOREACH(f ${MAXFLOW_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(MAXFLOW_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
     
    IF(DEFINED MAXFLOW_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-maxflow
                  ${MAXFLOW_SRC_FILES}
      )
      ADD_LIBRARY(external-library-maxflow-shared SHARED
                  ${MAXFLOW_SRC_FILES}
      )  
    ELSE(DEFINED MAXFLOW_SRC_FILES_EXIST)
        MESSAGE ("MAXFLOW not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED MAXFLOW_SRC_FILES_EXIST)
ENDIF()

IF(WITH_MAXFLOW_IBFS)
    SET(MAXFLOW_IBFS_SRC_FILES
        ${MAXFLOW_IBFS_PATCHEDSRCDIR}/ibfs.cpp
    )
    SET(MAXFLOW_IBFS_SRC_FILES_EXIST 1)
    FOREACH(f ${MAXFLOW_IBFS_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(MAXFLOW_IBFS_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
    
    IF(DEFINED MAXFLOW_IBFS_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-maxflow-ibfs 
                  ${MAXFLOW_IBFS_SRC_FILES}
      )
      ADD_LIBRARY(external-library-maxflow-ibfs-shared SHARED 
                  ${MAXFLOW_IBFS_SRC_FILES}
      )
    ELSE(DEFINED MAXFLOW_IBFS_SRC_FILES_EXIST)
        MESSAGE ("MAXFLOW_IBFS not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED MAXFLOW_IBFS_SRC_FILES_EXIST)
ENDIF()

IF(WITH_MRF)
    SET(MRF_SRC_FILES
        ${MRF_PATCHEDSRCDIR}/mrf.cpp
        ${MRF_PATCHEDSRCDIR}/ICM.cpp
        ${MRF_PATCHEDSRCDIR}/GCoptimization.cpp
        ${MRF_PATCHEDSRCDIR}/graph.cpp
        ${MRF_PATCHEDSRCDIR}/maxflow.cpp
        ${MRF_PATCHEDSRCDIR}/MaxProdBP.cpp
        ${MRF_PATCHEDSRCDIR}/LinkedBlockList.cpp
        ${MRF_PATCHEDSRCDIR}/regions-maxprod.cpp
        ${MRF_PATCHEDSRCDIR}/TRW-S.cpp
        ${MRF_PATCHEDSRCDIR}/BP-S.cpp
    )
    SET(MRF_SRC_FILES_EXIST 1)
    FOREACH(f ${MRF_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(MRF_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
     
    IF(DEFINED MRF_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-mrf
                  ${MRF_SRC_FILES}
      ) 
      ADD_LIBRARY(external-library-mrf-shared SHARED
                  ${MRF_SRC_FILES}
      )
    ELSE(DEFINED MRF_SRC_FILES_EXIST)
        MESSAGE ("MRF not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED MRF_SRC_FILES_EXIST)
ENDIF()
