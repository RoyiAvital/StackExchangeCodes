% StackExchange Signal Processing Q16227
% https://dsp.stackexchange.com/questions/16227
% Matching 2 Undirected Weighted Graph in MATLAB.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     01/07/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants


%% Parameters

dataFile = 'Data.csv';
numRows = 5;

costUnmatched = 1e6;


%% Generate / Load Data

mGG = readmatrix('Data.csv');
mG1 = mGG(1:numRows, :);
mG2 = mGG((numRows + 1):(2 * numRows), :);

vMRef = [2; 5; 4; 3; 1]; %<! By the OP (Seems to be not optimal)


%% Analysis

vU = diag(mG1);
vV = diag(mG2);

mC = -(vU * vV');

mM = matchpairs(mC, costUnmatched);
[~, vIdx] = sort(mM(:, 1), 1);
vM = mM(vIdx, 2);

modelCostRef = CalcCost(vU, vV, vMRef);
modelCost = CalcCost(vU, vV, vM);

% Brute Force Analysis
mAllPerm = perms(1:length(vU));

bestCost = 1e6;
jj = 0;
for ii = 1:size(mAllPerm, 1)
    currCost = CalcCost(vU, vV, mAllPerm(ii, :));
    if(currCost < bestCost)
        bestCost = currCost;
        jj = ii;
    end
end


%% Display Results


disp(['The OP matching cost              : ', num2str(modelCostRef)]);
disp(['The model matching cost           : ', num2str(modelCost)]);
disp(['The brute force matching best cost: ', num2str(bestCost)]);



%% Auxiliary Functions

function [ totalCost ] = CalcCost(vU, vV, vM)

totalCost = 0;

for ii = 1:length(vM)
    totalCost = totalCost + abs(vU(ii) - vV(vM(ii)));
end


end




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

