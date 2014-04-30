# MatLab links it's own version of hdf5 libraries and thus overides the dynamic 
# linked versions provided by the build system. This leads to a segfault in 
# MatLab if both hdf5 versions are different from each other.
# Solution: Static link HDF5 Library into the mexfiles
# (found at http://stackoverflow.com/questions/2188658/shared-library-locations-for-matlab-mex-files)
# TODO: Let cmake set correspondig CXX_FLAGS for static linking of the found hdf5 libraries.
# "-Wl,-Bdynamic -lstdc++" is required as libstdc++ does not support static linking into a dynamic library
#message("hdf5 libs: ${HDF5_LIBRARIES}") 
#hdf5 libs: /usr/lib/libhdf5.so;/usr/lib/libhdf5_hl.so;/usr/lib/libz.so
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /usr/lib/libhdf5.a /usr/lib/libhdf5_hl.a /usr/lib/libz.a")

#set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -t /usr/lib/libhdf5.a /usr/lib/libhdf5_hl.a /usr/lib/libz.a") 

if(WITH_MATLAB)
if(WITH_HDF5)
    if(WITH_OPENMP)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    endif()
    
    ADD_LIBRARY(handleCollector SHARED helper/handle/handleCollector.cpp)
    ADD_LIBRARY(loadModel MODULE model/loadModel.cpp)
    ADD_LIBRARY(storeModel MODULE model/storeModel.cpp)
    ADD_LIBRARY(newModel MODULE model/newModel.cpp)
    ADD_LIBRARY(deleteModel MODULE model/deleteModel.cpp)
    ADD_LIBRARY(addVariables MODULE model/addVariables.cpp)
    ADD_LIBRARY(addFactor MODULE model/addFactor.cpp)
    ADD_LIBRARY(addFactors MODULE model/addFactors.cpp)
    ADD_LIBRARY(addUnaries MODULE model/addUnaries.cpp) 
    ADD_LIBRARY(addPairwiseTerms MODULE model/addPairwiseTerms.cpp)
    ADD_LIBRARY(addFunction MODULE model/addFunction.cpp)
    ADD_LIBRARY(numFactors MODULE model/numFactors.cpp)
    ADD_LIBRARY(numLabels MODULE model/numLabels.cpp)
    ADD_LIBRARY(numVariables MODULE model/numVariables.cpp)
    ADD_LIBRARY(getFactorTable MODULE model/getFactorTable.cpp)
    ADD_LIBRARY(factorOrder MODULE model/factorOrder.cpp)
    ADD_LIBRARY(maxLabelOrder MODULE model/maxLabelOrder.cpp)
    ADD_LIBRARY(isGrid MODULE model/isGrid.cpp)
    ADD_LIBRARY(hasPottsFactor MODULE model/hasPottsFactor.cpp)
    ADD_LIBRARY(hasTruncatedAbsoluteDifferenceFactor MODULE model/hasTruncatedAbsoluteDifferenceFactor.cpp)
    ADD_LIBRARY(hasTruncatedSquaredDifferenceFactor MODULE model/hasTruncatedSquaredDifferenceFactor.cpp)
    ADD_LIBRARY(opengm MODULE interface/opengm.cpp)
    
    SET(MEXTARGETS loadModel
        storeModel
        newModel
        deleteModel
        addVariables
        addFactor
        addFactors
        addUnaries
        addPairwiseTerms
        addFunction
        numFactors
        numLabels
        numVariables
        getFactorTable
        factorOrder
        maxLabelOrder
        isGrid
        hasPottsFactor
        hasTruncatedAbsoluteDifferenceFactor
        hasTruncatedSquaredDifferenceFactor
        opengm
        # others...
    )

    SET_TARGET_PROPERTIES(${MEXTARGETS} PROPERTIES PREFIX "" SUFFIX ".${MATLAB_MEXEXT}")

    if(LINK_RT)
        find_library(RT rt)
        target_link_libraries(handleCollector rt)
        target_link_libraries(loadModel rt)
        target_link_libraries(storeModel rt)  
        target_link_libraries(newModel rt)
        target_link_libraries(deleteModel rt)  
        target_link_libraries(addVariables rt)  
        target_link_libraries(addFactor rt)
        target_link_libraries(addFactors rt)
        target_link_libraries(addUnaries rt) 
        target_link_libraries(addPairwiseTerms rt)
        target_link_libraries(addFunction rt)
        target_link_libraries(numFactors rt)
        target_link_libraries(numLabels rt)
        target_link_libraries(numVariables rt)
        target_link_libraries(getFactorTable rt)
        target_link_libraries(factorOrder rt)
        target_link_libraries(maxLabelOrder rt)
        target_link_libraries(isGrid rt)
        target_link_libraries(hasPottsFactor rt)
        target_link_libraries(hasTruncatedAbsoluteDifferenceFactor rt)
        target_link_libraries(hasTruncatedSquaredDifferenceFactor rt)
        target_link_libraries(opengm rt)
    endif(LINK_RT)

    target_link_libraries(loadModel handleCollector)
    target_link_libraries(storeModel handleCollector)  
    target_link_libraries(newModel handleCollector)
    target_link_libraries(deleteModel handleCollector)  
    target_link_libraries(addVariables handleCollector) 
    target_link_libraries(addFactor handleCollector)
    target_link_libraries(addFactors handleCollector)
    target_link_libraries(addUnaries handleCollector)
    target_link_libraries(addPairwiseTerms handleCollector)
    target_link_libraries(addFunction handleCollector)   
    target_link_libraries(numFactors handleCollector)
    target_link_libraries(numLabels handleCollector)
    target_link_libraries(numVariables handleCollector)
    target_link_libraries(getFactorTable handleCollector)
    target_link_libraries(factorOrder handleCollector)
    target_link_libraries(maxLabelOrder handleCollector)
    target_link_libraries(isGrid handleCollector)
    target_link_libraries(hasPottsFactor handleCollector)
    target_link_libraries(hasTruncatedAbsoluteDifferenceFactor handleCollector)
    target_link_libraries(hasTruncatedSquaredDifferenceFactor handleCollector)
    target_link_libraries(opengm handleCollector)
        
    if(WITH_MAXFLOW)
        target_link_libraries(opengm external-library-maxflow-shared)
    endif(WITH_MAXFLOW)
    if(WITH_CPLEX)
        if(WIN32)
           target_link_libraries(opengm wsock32.lib ${CPLEX_LIBRARIES})
        else()
           target_link_libraries(opengm ${CMAKE_THREAD_LIBS_INIT} ${CPLEX_LIBRARIES})
        endif(WIN32)
    endif(WITH_CPLEX)

    if(WITH_QPBO)
           target_link_libraries(opengm external-library-qpbo-shared)
    endif(WITH_QPBO)

    if(WITH_CONICBUNDLE)
      target_link_libraries(opengm ${CONICBUNDLE_LIBRARY})
    endif(WITH_CONICBUNDLE)
    
    if(WITH_MRF)
      target_link_libraries(opengm external-library-mrf-shared)
    endif(WITH_MRF)
    
    if(WITH_GCO)
      target_link_libraries(opengm external-library-gco)
    endif(WITH_GCO)
    
    if(WITH_FASTPD)
      target_link_libraries(opengm external-library-fastpd) 
    endif(WITH_FASTPD)
    
    if(WITH_GRANTE)
        target_link_libraries(opengm external-library-grante)
    endif(WITH_GRANTE)

    ## INSTALL
   INSTALL_TARGETS(/matlab/mex handleCollector) 
   INSTALL_TARGETS(/matlab/mex loadModel) 
   INSTALL_TARGETS(/matlab/mex storeModel) 
   INSTALL_TARGETS(/matlab/mex deleteModel) 
   INSTALL_TARGETS(/matlab/mex addVariables) 
   INSTALL_TARGETS(/matlab/mex addFactor) 
   INSTALL_TARGETS(/matlab/mex addFactors) 
   INSTALL_TARGETS(/matlab/mex addUnaries)  
   INSTALL_TARGETS(/matlab/mex addPairwiseTerms) 
   INSTALL_TARGETS(/matlab/mex addFunction) 
   INSTALL_TARGETS(/matlab/mex numFactors) 
   INSTALL_TARGETS(/matlab/mex numLabels) 
   INSTALL_TARGETS(/matlab/mex numVariables) 
   INSTALL_TARGETS(/matlab/mex getFactorTable) 
   INSTALL_TARGETS(/matlab/mex factorOrder) 
   INSTALL_TARGETS(/matlab/mex maxLabelOrder) 
   INSTALL_TARGETS(/matlab/mex isGrid) 
   INSTALL_TARGETS(/matlab/mex hasPottsFactor) 
   INSTALL_TARGETS(/matlab/mex hasTruncatedAbsoluteDifferenceFactor) 
   INSTALL_TARGETS(/matlab/mex hasTruncatedSquaredDifferenceFactor) 
   INSTALL_TARGETS(/matlab/mex opengm) 

else(WITH_HDF5)
  message(WARNING,"Matlab interface requires to activate WITH_HDF5")
endif(WITH_HDF5)
else(WITH_MATLAB)
   message(WARNING,"Matlab interface requires to activate WITH_MATLAB")
endif(WITH_MATLAB) 

