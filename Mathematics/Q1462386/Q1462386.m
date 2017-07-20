% Mathematics Q1462386
% https://math.stackexchange.com/questions/1462386
% Solving Non Linear System of Equations with Matlab
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     20/07/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;

dimOrder        = 4;
numIterations   = 15;
difMode         = DIFF_MODE_CENTRAL;
epsVal          = 1e-6;


%% Generate Data

mA = randn([dimOrder, dimOrder]);
vX = randn([dimOrder, 1]);

vB = (mA * vX) + exp(vX);

hObjFun     = @(vX) (mA * vX) + exp(vX) - vB;
hObjFunI    = @(vX, ii) (mA(ii, :) * vX) + exp(vX(ii)) - vB(ii);
hJcobFun    = @(vX) mA + diag(exp(vX)); %<! Jacobian (Transpose of Gradient)

vCostFunA = zeros([numIterations, 1]); %<! Analytic
vCostFunN = zeros([numIterations, 1]); %<! Numerical


%% Run Analysis

% Analytic Analysis
vXX         = zeros([dimOrder, 1]);
vCostFunA(1) = norm(hObjFun(vXX));

for ii = 2:numIterations
    vXX = vXX - (hJcobFun(vXX) \ hObjFun(vXX));
    % vXX = vXX - (pinv(hJcobFun(vXX)) * hObjFun(vXX));
    
    vCostFunA(ii) = norm(hObjFun(vXX));
end

% Numeric Analysis

vXX         = zeros([dimOrder, 1]);
mJ          = zeros([dimOrder, dimOrder]);
vCostFunN(1) = norm(hObjFun(vXX));

for ii = 2:numIterations
    
    for jj = 1:dimOrder
        hObjFunJ = @(vX) hObjFunI(vX, jj);
        mJ(jj, :) = CalcFunGrad(vXX, hObjFunJ, difMode, epsVal).';
    end
    
    vXX = vXX - (mJ \ hObjFun(vXX));
    % vXX = vXX - (pinv(hJcobFun(vXX)) * hObjFun(vXX));
    
    vCostFunN(ii) = norm(hObjFun(vXX));
end


%% Display Results

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [vCostFunA, vCostFunN]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Newton''s Method Iterations Cost Function'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number [n]', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Cost Function', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['Analytic Jacobian'], ['Numerical Jacobian']});


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

