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


IF(WITH_GCO)
    SET(GCO_SRC_FILES
        ${GCO_PATCHEDSRCDIR}/GCoptimization.cpp
        ${GCO_PATCHEDSRCDIR}/graph.cpp
        ${GCO_PATCHEDSRCDIR}/LinkedBlockList.cpp
        ${GCO_PATCHEDSRCDIR}/maxflow.cpp
    )
    
    SET(GCO_SRC_FILES_EXIST 1)
    FOREACH(f ${GCO_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(GCO_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
     
    IF(DEFINED GCO_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-gco
                  ${GCO_SRC_FILES}
      ) 
      ADD_LIBRARY(external-library-gco-shared SHARED
                  ${GCO_SRC_FILES}
      )
    ELSE(DEFINED GCO_SRC_FILES_EXIST)
        MESSAGE ("GCO not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED GCO_SRC_FILES_EXIST)
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

IF(WITH_FASTPD)
    SET(FASTPD_SRC_FILES
        ${FASTPD_PATCHEDSRCDIR}/graph.cpp
        ${FASTPD_PATCHEDSRCDIR}/LinkedBlockList.cpp
        ${FASTPD_PATCHEDSRCDIR}/maxflow.cpp
    )
    SET(FASTPD_SRC_FILES_EXIST 1)
    FOREACH(f ${FASTPD_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(FASTPD_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
    
    IF(DEFINED FASTPD_SRC_FILES_EXIST)    
      ADD_LIBRARY(external-library-fastpd
                  ${FASTPD_SRC_FILES}
      ) 
      ADD_LIBRARY(external-library-fastpd-shared SHARED
                  ${FASTPD_SRC_FILES}
      )
    ELSE(DEFINED FASTPD_SRC_FILES_EXIST)
        MESSAGE ("FASTPD not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED FASTPD_SRC_FILES_EXIST)
ENDIF()

IF(WITH_GRANTE)
    SET(GRANTE_SRC_FILES
        ${GRANTE_PATCHEDSRCDIR}/Factor.cpp
        ${GRANTE_PATCHEDSRCDIR}/FactorGraph.cpp
        ${GRANTE_PATCHEDSRCDIR}/FactorGraphModel.cpp
        ${GRANTE_PATCHEDSRCDIR}/FactorType.cpp        
        ${GRANTE_PATCHEDSRCDIR}/FactorGraphStructurizer.cpp
        ${GRANTE_PATCHEDSRCDIR}/DisjointSet.cpp
        ${GRANTE_PATCHEDSRCDIR}/LogSumExp.cpp
        ${GRANTE_PATCHEDSRCDIR}/TreeInference.cpp        
        ${GRANTE_PATCHEDSRCDIR}/GibbsSampler.cpp
        ${GRANTE_PATCHEDSRCDIR}/FunctionMinimization.cpp
        ${GRANTE_PATCHEDSRCDIR}/FunctionMinimizationProblem.cpp
        ${GRANTE_PATCHEDSRCDIR}/Likelihood.cpp        
        ${GRANTE_PATCHEDSRCDIR}/InferenceMethod.cpp
        ${GRANTE_PATCHEDSRCDIR}/ParameterEstimationMethod.cpp
        ${GRANTE_PATCHEDSRCDIR}/MaximumLikelihood.cpp
        ${GRANTE_PATCHEDSRCDIR}/FactorGraphObservation.cpp        
        ${GRANTE_PATCHEDSRCDIR}/Prior.cpp
        ${GRANTE_PATCHEDSRCDIR}/NormalPrior.cpp
        ${GRANTE_PATCHEDSRCDIR}/LaplacePrior.cpp
        ${GRANTE_PATCHEDSRCDIR}/MaximumPseudolikelihood.cpp        
        ${GRANTE_PATCHEDSRCDIR}/FactorGraphUtility.cpp
        ${GRANTE_PATCHEDSRCDIR}/Pseudolikelihood.cpp
        ${GRANTE_PATCHEDSRCDIR}/StudentTPrior.cpp
        ${GRANTE_PATCHEDSRCDIR}/StructuredLossFunction.cpp        
        ${GRANTE_PATCHEDSRCDIR}/StructuredHammingLoss.cpp
        ${GRANTE_PATCHEDSRCDIR}/StructuredSVM.cpp
        ${GRANTE_PATCHEDSRCDIR}/StochasticFunctionMinimizationProblem.cpp
        ${GRANTE_PATCHEDSRCDIR}/StochasticFunctionMinimization.cpp        
        ${GRANTE_PATCHEDSRCDIR}/StructuredPerceptron.cpp
        ${GRANTE_PATCHEDSRCDIR}/RandomFactorGraphGenerator.cpp
        ${GRANTE_PATCHEDSRCDIR}/VAcyclicDecomposition.cpp
        ${GRANTE_PATCHEDSRCDIR}/SubFactorGraph.cpp        
        ${GRANTE_PATCHEDSRCDIR}/FactorConditioningTable.cpp
        ${GRANTE_PATCHEDSRCDIR}/ConditionedFactorType.cpp
        ${GRANTE_PATCHEDSRCDIR}/Conditioning.cpp
        ${GRANTE_PATCHEDSRCDIR}/MaximumCompositeLikelihood.cpp        
        ${GRANTE_PATCHEDSRCDIR}/TreeCoverDecomposition.cpp
        ${GRANTE_PATCHEDSRCDIR}/LinearProgrammingMAPInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/FactorDataSource.cpp
        ${GRANTE_PATCHEDSRCDIR}/FactorGraphPartialObservation.cpp        
        ${GRANTE_PATCHEDSRCDIR}/BeliefPropagation.cpp
        ${GRANTE_PATCHEDSRCDIR}/RBFNetworkRegression.cpp
        ${GRANTE_PATCHEDSRCDIR}/RBFNetwork.cpp
        ${GRANTE_PATCHEDSRCDIR}/NonlinearRBFFactorType.cpp        
        ${GRANTE_PATCHEDSRCDIR}/GibbsInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/StructuredMeanFieldInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/ExpectationMaximization.cpp
        ${GRANTE_PATCHEDSRCDIR}/RandomSource.cpp
        ${GRANTE_PATCHEDSRCDIR}/AISInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/SimulatedAnnealingInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/MaximumCrissCrossLikelihood.cpp
        ${GRANTE_PATCHEDSRCDIR}/DiffusionInference.cpp        
        ${GRANTE_PATCHEDSRCDIR}/BruteForceExactInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/LinearFactorType.cpp
        ${GRANTE_PATCHEDSRCDIR}/NaivePiecewiseTraining.cpp
        ${GRANTE_PATCHEDSRCDIR}/ContrastiveDivergence.cpp
        ${GRANTE_PATCHEDSRCDIR}/ContrastiveDivergenceTraining.cpp
        ${GRANTE_PATCHEDSRCDIR}/HyperbolicPrior.cpp
        ${GRANTE_PATCHEDSRCDIR}/CompositeMinimization.cpp
        ${GRANTE_PATCHEDSRCDIR}/CompositeMinimizationProblem.cpp        
        ${GRANTE_PATCHEDSRCDIR}/SwendsenWangSampler.cpp
        ${GRANTE_PATCHEDSRCDIR}/SwendsenWangInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/ParallelTemperingInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/SAMCInference.cpp        
        ${GRANTE_PATCHEDSRCDIR}/TestModels.cpp
        ${GRANTE_PATCHEDSRCDIR}/NaiveMeanFieldInference.cpp
        ${GRANTE_PATCHEDSRCDIR}/MaximumTreePseudoLikelihood.cpp
        ${GRANTE_PATCHEDSRCDIR}/DisjointSetBT.cpp        
        ${GRANTE_PATCHEDSRCDIR}/MultichainGibbsInference.cpp
    )
    
    SET(GRANTE_SRC_FILES_EXIST 1)
    FOREACH(f ${GRANTE_SRC_FILES})
        IF(NOT EXISTS ${f})
            UNSET(GRANTE_SRC_FILES_EXIST)
        ENDIF(NOT EXISTS ${f})
    ENDFOREACH(f)
    
    IF(DEFINED GRANTE_SRC_FILES_EXIST)
      add_definitions(-fopenmp -fno-omit-frame-pointer -std=c++0x)    
      ADD_LIBRARY(external-library-grante
                  ${GRANTE_SRC_FILES}
      )
    ELSE(DEFINED GRANTE_SRC_FILES_EXIST)
        MESSAGE ("Grante not installed, run make externalLibs first and configure again")
    ENDIF(DEFINED GRANTE_SRC_FILES_EXIST)
ENDIF()
