clear all;

% add opengm interface to path
addpath('../../');

% add model to path
addpath('../model/');

% add functions to path
addpath('../model/functions');

% create model
disp('create model');
gm = openGMModel;

% num variables
disp('number of variables: ');
disp(gm.numberOfVariables());

% add variables
disp('add variables');
gm.addVariables([2,2,2,2]);

% num variables
disp('number of variables: ');
disp(gm.numberOfVariables());

% num labels
disp('number of labels of variable 2: ');
disp(gm.numberOfLabels(2));

% create first oder function
disp('create first oder function');
A = [1, 2];
firstOrderFunction = openGMExplicitFunction(2, A);
clear A;

% create second order function
disp('create second order function');
B = [1, 2; 3, 4];
secondOrderFunction = openGMExplicitFunction([2, 2], B);
clear B;

% create third order function
disp('create third order function');
C = cat(3, [1, 2; 3, 4], [5, 6; 7, 8]);
thirdOrderFunction = openGMExplicitFunction([2, 2, 2], C);
clear C;

% create fourth order function
disp('create fourth order function');
D = cat(4, cat(3, [1, 2; 3, 4], [5, 6; 7, 8]), cat(3, [9, 10; 11, 12], [13, 14; 15, 16]));
fourthOrderFunction = openGMExplicitFunction([2, 2, 2, 2], D);
clear D;

% create potts function
disp('create potts function');
pottsFunction = openGMPottsFunction([2, 2], 100, 1);

% create pottsN function
disp('create pottsN function');
pottsNFunction = openGMPottsNFunction([2, 2, 2, 2], 1, 1000);

% create pottsG function
disp('create pottsG function');
values = [10000, 20000, 30000, 40000, 50000];
pottsGFunction = openGMPottsGFunction([2, 2, 2], values);

% create tl2 function
disp('create tl2 function');
tl2Function = openGMTruncatedSquaredDifferenceFunction([2, 2], 100000, 200000);

% create tl1 function
disp('create tl1 function');
tl1Function = openGMTruncatedAbsoluteDifferenceFunction([2, 2], 1000000, 2000000);

% add functions
disp('add functions');

gm.addFunction(firstOrderFunction);
gm.addFunction(secondOrderFunction);
gm.addFunction(thirdOrderFunction);
gm.addFunction(fourthOrderFunction);

gm.addFunction(pottsFunction);
gm.addFunction(pottsNFunction);
gm.addFunction(pottsGFunction);
gm.addFunction(tl2Function);
gm.addFunction(tl1Function);

% num factors
disp('number of factors: ');
disp(gm.numberOfFactors());

% add first oder factors
disp('add first oder factors');
gm.addFactor(firstOrderFunction, 0);
gm.addFactor(firstOrderFunction, 1);
gm.addFactor(firstOrderFunction, 2);
gm.addFactor(firstOrderFunction, 3);

% equivalent
%gm.addFactors(firstOrderFunction, [0, 1, 2, 3]);

% add second order factors
disp('add second order factors');
variables = cat(2, [0, 1]', [0, 2]', [1, 3]', [2, 3]');
gm.addFactors(secondOrderFunction, variables);

% equivalent
% gm.addFactor(secondOrderFunction, [0, 1]);
% gm.addFactor(secondOrderFunction, [0, 2]);
% gm.addFactor(secondOrderFunction, [1, 3]);
% gm.addFactor(secondOrderFunction, [2, 3]);

% add third order factors
disp('add third order factors');
variables = cat(2, [0, 1, 2]', [1, 2, 3]');
gm.addFactors(thirdOrderFunction, variables);

% equivalent
% gm.addFactor(thirdOrderFunction, [0, 1, 2]);
% gm.addFactor(thirdOrderFunction, [1, 2, 3]);

% add fourth order factor
disp('add fourth order factor');
gm.addFactors(fourthOrderFunction, [0, 1, 2, 3]');

% equivalent
%gm.addFactor(fourthOrderFunction, [0, 1, 2, 3]);

% add potts factor
disp('add potts factor');
gm.addFactor(pottsFunction, [0, 3]);

% add pottsN factor
disp('add pottsN factor');
gm.addFactor(pottsNFunction, [0, 1, 2, 3]);

% add pottsG factor
disp('add pottsG factor');
gm.addFactor(pottsGFunction, [1, 2, 3]);

% add tl2 factor
disp('add tl2 factor');
gm.addFactor(tl2Function, [1, 3]);

% add tl1 factor
disp('add tl1 factor');
gm.addFactor(tl1Function, [0, 1]);

% num factors
disp('number of factors: ');
disp(gm.numberOfFactors());

% get factor table of factor 0
[factor0Table, factor0Variables] = gm.getFactorTable(0);
disp('factor table of factor 0: ');
disp(factor0Table)
disp('variables of factor 0: ');
disp(factor0Variables)

% get factor table of factor 4
[factor4Table, factor4Variables] = gm.getFactorTable(4);
disp('factor table of factor 4: ');
disp(factor4Table)
disp('variables of factor 4: ');
disp(factor4Variables)

% get factor table of factor 8
[factor8Table, factor8Variables] = gm.getFactorTable(8);
disp('factor table of factor 8: ');
disp(factor8Table)
disp('variables of factor 8: ');
disp(factor8Variables)

% get factor table of factor 11 (potts factor)
[factor11Table, factor11Variables] = gm.getFactorTable(11);
disp('factor table of factor 11 (potts factor): ');
disp(factor11Table)
disp('variables of factor 11: ');
disp(factor11Variables)

% get factor table of factor 12 (pottsN factor)
[factor12Table, factor12Variables] = gm.getFactorTable(12);
disp('factor table of factor 12 (pottsN factor): ');
disp(factor12Table)
disp('variables of factor 12: ');
disp(factor12Variables)

% get factor table of factor 13 (pottsG factor)
[factor13Table, factor13Variables] = gm.getFactorTable(13);
disp('factor table of factor 13 (pottsG factor): ');
disp(factor13Table)
disp('variables of factor 13: ');
disp(factor13Variables)

% print model informations
disp('print model informations');
opengm('modelinfo', 'm', gm);

% running inference and store results in hdf5 file res.h5
opengm('model', gm, 'a', 'TRBP', 'o', 'res.h5', 'p', 1, 'v', 'maxIt', 3, 'bound', 0.01);

% running inference and return results to MatLab
inferenceResults = opengm('model', gm, 'a', 'TRBP', 'p', 1, 'v', 'maxIt', 3, 'bound', 0.01);

% evaluate model
disp('evaluate model');
% if no return paramter is specified evaluate will print the computed
% result on screen
opengm('evaluate', [1, 0, 1, 1], 'm', gm);
% otherwise it will return the computed result to MatLab for further
% processing
result = opengm('evaluate', [1, 0, 1, 1], 'm', gm);
disp('result: ');
disp(result);

% store model
disp('store model')
gm.store('testmodel.h5', 'gm')
