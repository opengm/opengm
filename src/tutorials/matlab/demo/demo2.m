function L = demo2(I,color, T, lambda) 
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
    fastUnaries = 1;
    if(fastUnaries)
        gm.addUnaries(0:numVariables-1, [d(:)';T*ones(1,numVariables)]);  
    else
       disp('please wait for a few minutes :-)');
       progress = 0;
       for i=1:numVariables
           if(floor(i/numVariables*100)>progress)
                progress = floor(i/numVariables*100);
                disp([num2str(progress),' %']);
           end
          unaryFunction = openGMExplicitFunction(numLabels, [d(i);T]);
          gm.addFunction(unaryFunction);
          gm.addFactor(unaryFunction, i-1);
       end
    end
    
    % add potts functions
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

    % print model informations
    disp('print model informations');
    opengm('modelinfo', 'm', gm);
    
    %infer
    disp('start inference');
    opengm('a','GRAPHCUT', 'm', gm, 'o','out.h5');
    disp('load result');
    x = h5read('out.h5','/states');
    L = uint8(reshape(1-x,size(I,1),size(I,2))*255);
end