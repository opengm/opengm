function [ L ] = binaryNDSeg( U, lambda )
% binary segmentation on a nD-grid with a Issing model
%   Detailed explanation goes here
    dims      = size(U);
    numStates = dims(1);
    shape     = dims(2:end); 
    numVar    = prod(shape); 
    
    opengmpath      = '~/GIT/opengm-2.3/opengm/';
    opengmbuildpath = '~/GIT/opengm-2.3/opengm/build/';

    addpath( [opengmbuildpath,'src/interfaces/matlab/opengm/mex-src/'])
    addpath(genpath([opengmpath,'src/interfaces/matlab/opengm/m_files/model/']))
    
    gm = openGMModel;
    gm.addVariables(ones(1,numVar)*numStates);  
    
    %% Add Unaries
    gm.addUnaries( 0:(numVar-1),U(:,:) );
    
    
    %% Add Regularizer
    pottsFunction = openGMPottsFunction([numStates, numStates], 0, lambda);
    gm.addFunction(pottsFunction);
    
    if(numel(shape)==2)
        % 1st dim
        variables = 0 : (numVar - 1); 
        ind = sub2ind(shape,ones(1,shape(2))*shape(1),1:shape(2));
        variables(ind) = []; 
        variables = cat(1, variables, variables + 1);
        gm.addFactors(pottsFunction, variables); 
        % 2st dim
        variables = 0 : (numVar - 1); 
        ind = sub2ind(shape,1:shape(1), ones(1,shape(1))*shape(2));
        variables(ind) = []; 
        variables = cat(1, variables, variables + shape(1));
        gm.addFactors(pottsFunction, variables); 
    elseif(numel(shape)==3)
         V = reshape(0 : (numVar - 1),shape);
         %1
         T = V(1:end-1,:,:);
         gm.addFactors(pottsFunction, cat(1, T(:)', T(:)'+1)); 
         %2
         T = V(:,1:end-1,:);
         gm.addFactors(pottsFunction, cat(1, T(:)', T(:)'+shape(1))); 
         %3
         T = V(:,:,1:end-1);
         gm.addFactors(pottsFunction, cat(1, T(:)', T(:)'+prod(shape(1:2))));           
    else
        disp('Other than 2 dim need to be implemented')
        
    end
    opengm('m',gm,'a','GRAPHCUT','o','out.h5');
    L=reshape(h5read('out.h5','/states'),shape);
end

