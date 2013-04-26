% create a grid model and measure performance
clear all;

% add opengm interface to path
addpath('../../');

% add model to path
addpath('../model/');

% add functions to path
addpath('../model/functions');

% parameter
numVariablesN = 300; % number of variables of first dimension
numVariablesM = 300; % number of variables of second dimension
numLabels = 10;      % number of labels for each variable

tic

numVariables = numVariablesN * numVariablesM;
% create functions
% unary function
unaryFunction = openGMExplicitFunction(numLabels, rand(1, numLabels));

% binary function
pottsFunction = openGMPottsFunction([numLabels, numLabels], 0, 1);

% create model
gm = openGMModel;

% add variables
gm.addVariables(repmat(numLabels, 1, numVariables));

% add functions
gm.addFunction(unaryFunction);
gm.addFunction(pottsFunction);

% add unary factor to each variable
variables = 0 : (numVariables - 1);
gm.addFactors(unaryFunction, variables);

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

toc

% print model informations
disp('print model informations');
opengm('modelinfo', 'm', gm);
