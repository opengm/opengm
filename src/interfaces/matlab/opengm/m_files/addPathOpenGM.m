function addPathOpenGM( openGMSourceDir, openGMBuildDir )
%ADDPATHOPENGM Adds all openGM folders required by MatLab-Interface to path
%   Detailed explanation goes here
    

    % add opengm model to path
    relativeModelPath = 'src/interfaces/matlab/opengm/m_files/model/';
    addpath([openGMSourceDir, relativeModelPath]);
    
    % add opengm functions to path
    relativeFunctionsPath = 'src/interfaces/matlab/opengm/m_files/model/functions/';
    addpath([openGMSourceDir, relativeFunctionsPath]);
    
    % add opengm mex files to path
    relativeMexFilesPath = 'src/interfaces/matlab/opengm/mex-src/';
    addpath([openGMBuildDir, relativeMexFilesPath]);    
end

