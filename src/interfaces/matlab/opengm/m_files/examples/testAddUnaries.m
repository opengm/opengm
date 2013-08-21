clear all;

% create model
disp('create model');
gm = openGMModel;

% add variables
disp('add variables');
gm.addVariables([2,2,2,2,5]);

% add unaries
disp('add unaries');

% add factor [1, 2] to variable 0, factor [3, 4] to variable 1 and factor [5, 6]
% to variable 2
gm.addUnaries(0:2, [[1,2]', [3,4]', [5,6]']);

% add factor [7, 8] to variable 3 and factor [9, 10, 11, 12, 13] to variable 4
doNotCare = NaN;
gm.addUnaries(3:4, [[7,8,doNotCare,doNotCare,doNotCare]', [9,10,11,12,13]']);

% print model info
disp('printing model info');
opengm('modelinfo', 'm', gm);