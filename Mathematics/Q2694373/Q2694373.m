% Mathematics Q2694373
% https://math.stackexchange.com/questions/2694373
% Generalized Projection of a Matrix on the Non Negative Orthant
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     17/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

numRows = 2;
numCols = 3;

numIterations   = 1000;
stepSize        = 0.0075;


%% Generate Data

mZ = randn([numRows, numCols]);
mH = randn([numCols, numCols]);
mH = (mH.' * mH) + (0.5 * eye(numCols));

mL = chol(mH, 'lower');

hObjFun = @(mY) trace((mY - mZ) * mH * (mY - mZ).');

% Sanity Check
% See http://ask.cvxr.com/t/148
% mZL = mZ * mL;
% abs(trace(mZ * mH * mZ.') - sum(mZL(:) .^ 2))


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable mY(numRows, numCols);
    minimize( sum_square( reshape((mY - mZ) * mL, [numRows * numCols, 1]) ) );
    subject to
        mY >= 0;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mY(:).'), ' ]']);
disp([' ']);


%% Solution by Projected Gradient Method

vObjValPgd = zeros([numIterations, 1]);

mY = zeros([numRows, numCols]);
vObjValPgd(1) = hObjFun(mY);

for ii = 2:numIterations
    % vG = ((mY - mZ) * mH.') + ((mY - mZ) * mH);
    vG = (mY - mZ) * (mH.' + mH);
    mY = mY - (stepSize * vG);
    
    mY = max(mY, 0); %<! Projection onto Non Negative Orthant
    
    vObjValPgd(ii) = hObjFun(mY); 
end

disp([' ']);
disp(['Projected Gradient Descent Method Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjValPgd(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mY(:).'), ' ]']);
disp([' ']);


%% Display Reesults

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot([1:numIterations], [cvx_optval * ones([numIterations, 1]), vObjValPgd]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(2:end), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', ['Generalized Projection of a Matrix on the Non Negative Orthant'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['CVX'], ['Projected Gradient Descent']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

