function L = demo3(I,color, T, lambda) 
    I = double(I);
    d = abs(I(:,:,1)-color(1)) +  abs(I(:,:,2)-color(2)) +  abs(I(:,:,3)-color(3));

    % parameter
    numVariablesN = size(I,1); % number of variables of first dimension
    numVariablesM = size(I,2); % number of variables of second dimension
    numLabels     = 2;      % number of labels for each variable


    numVariables = numVariablesN * numVariablesM;
 
    % binary function
    pottsFunction = openGMPottsFunction([numLabels, numLabels], 0, lambda);

    % create model
    gm = openGMModel;

    % add variables
    gm.addVariables(repmat(numLabels, 1, numVariables));

    % add unary functios and factor to each variable
    gm.addUnaries(0:numVariables-1, [d(:)';T*ones(1,numVariables)]);  
 
    % add functions
    gm.addFunction(pottsFunction);

    % add binary factors to create grid structure
    % horizontal factors
    variablesH = 0 : (numVariables - 1);
    variablesH(numVariablesN : numVariablesN : numVariables) = [];
    variablesH = cat(1, variablesH, variablesH + 1);
    % vertical factors
    variablesV = 0 : (numVariables - (numVariablesN + 1));
    variablesV = cat(1, variablesV, variablesV + numVariablesN);
    % concatenate horizontal and vertical factors
    variables = cat(2, variablesH, variablesV);
    % add factors
    gm.addFactors(pottsFunction, variables);

    % store model into file 'gmfile.h5' with datasetname 'gm'
    gm.store('gmfile.h5','gm');
end