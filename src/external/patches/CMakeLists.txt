
set(externalLibs_scripts
    MaxFlow/patchMaxFlow-v3.02.sh
    MRF/patchMRF-v2.1.sh
)

add_custom_target(externalLibs
                  COMMENT "Download, patch and install external libraries used by openGM"
)

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchAD3.sh
                   COMMENT "AD3"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/AD3"
)


add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchMRF-v2.1.sh
                   COMMENT "MRF"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MRF"
)

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchGCO.sh
                   COMMENT "GCO"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/GCO"
)

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchMaxFlow-v3.02.sh
                   COMMENT "MaxFlow"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MaxFlow"
)   

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchQPBO-v1.3.sh
                   COMMENT "QPBO"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/QPBO"
)

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchTRWS-v1.3.sh
                   COMMENT "TRWS"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/TRWS"
)

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchConicBundle-v0.3.11.sh
                   COMMENT "ConicBundle"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ConicBundle"
)
  
add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchIBFS.sh
                   COMMENT "IBFS"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/IBFS"
)  
                           
add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchFastPD.sh ARGS ${FASTPD_URL}
                   COMMENT "FastPD"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/FastPD"
)   

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchGrante-v1.0.sh
                   COMMENT "Grante"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/grante"
)                 

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchDAOOPT.sh
                   COMMENT "DAOOPT"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/DAOOPT"
) 

add_custom_command(TARGET externalLibs PRE_BUILD                   
                   COMMAND ./patchmplp_ver2.sh
                   COMMENT "MPLP"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MPLP"
) 