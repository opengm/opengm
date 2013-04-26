% add opengm interface to path
addpath('../../');

% add model to path
addpath('../model/');

disp('creating empty model');
gm = openGMModel;
disp('printing model info');
opengm('modelinfo', 'm', gm);
disp('adding variables');
gm.addVariables([2,1,3,5,2,2]);
disp('printing model info again');
opengm('modelinfo', 'm', gm);
disp('clearing all');
clear all;
%exit;