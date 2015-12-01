


H=80;
W=100;
gm = openGMModel;

I=zeros(H,W);
I(40:60,40:60)=1;
I=I+rand(size(I))*0.8;

% add variables
gm.addVariables([ ones(1,H*W)*2 ]); 
gm.addUnaries(0:H*W-1, [I(:)';1-I(:)']);  

numVariablesH =W;
numVariablesV =H;

  numVariables = numVariablesH * numVariablesV;
    % horizontal factors
    variablesH = 0 : (numVariables - 1);
    variablesH(numVariablesH : numVariablesH : numVariables) = [];
    variablesH = cat(1, variablesH, variablesH + 1);
    % vertical factors
    variablesV = 0 : (numVariables - (numVariablesV + 1));
    variablesV = cat(1, variablesV, variablesV + numVariablesV);
    % concatenate horizontal and vertical factors
    variablePairs = cat(2, variablesH, variablesV);

  %gm.addPairwiseTerms(variablePairs , [0;1;1;0]*ones(1,numel(variablePairs)/2));



  gm.store('gm.h5','gm')

 %infer
 disp('start inference');
 opengm('a','GRAPHCUT', 'm', gm, 'o','out.h5');
 disp('load result');
 x = h5read('out.h5','/states');
 L = uint8(reshape(1-x,H,W)*255);