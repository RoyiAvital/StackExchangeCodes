% Mathematics Q1781856
% https://math.stackexchange.com/questions/1781856
% Least Square Linear Regression with Elements Wise Magnitude Equality
% constraints
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     09/09/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;


%% Simulation Parameters

numRows     = 10;
numCols     = 5;
paramLambda = 0.05;
paramLambdaMax = 55;

difMode     = DIFF_MODE_CENTRAL;
epsVal      = 1e-6;


optVal          = inf;
stepSize        = 0.0025;
numIterations   = 500;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);

vA = 10 * rand([numCols, 1]);

% See https://stackoverflow.com/questions/41229595
hDecToBin = @(inputNum, numBits) (rem(floor(inputNum(:) * pow2(1 - numBits:0)), 2) == 1);


%% Validate Derivative

vX         = randn([numCols, 1]);
hRegFun    = @(vX) sum((abs(vX) - vA) .^ 2); %<! Regularization Function

vGNumerical = CalcFunGrad(vX, hRegFun, difMode, epsVal);

% mX = mX * ((1 / paramP) * (sum(abs(vX) .^ paramP) .^ ((1 / paramP) - 1)));
% vGAnalytic = paramP * mX * vX;

vGAnalytic  = 2 * sign(vX) .* (abs(vX) - vA);

disp(['Maximum Deviation Between Analytic and Numerical Derivative - ', num2str( max(abs(vGNumerical - vGAnalytic)) )]);


%% Solution by Brute Force

numCombinations = 2 ^ numCols; %<! Number of Combinations

for ii = 1:numCombinations
    vBinaryFlag = double(hDecToBin(ii - 1, numCols));
    vBinaryFlag(vBinaryFlag == 0) = -1;
    vX = vBinaryFlag(:) .* vA;
    
    objVal = 0.5 * sum(((mA * vX) - vB) .^ 2);
    if(objVal <= optVal)
        optVal = objVal;
        vXOpt = vX;
    end
    
end

disp([' ']);
disp(['Brute Force Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(optVal)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vXOpt.'), ' ]']);
disp([' ']);


%% Solution by Gradient Descent

hObjFun     = @(vX) (0.5 * sum((mA * vX - vB) .^ 2)) + (paramLambda * sum((abs(vX) - vA) .^ 2));
hObjFunLs   = @(vX) (0.5 * sum((mA * vX - vB) .^ 2));

vObjVal     = zeros([numIterations, 1]);
vObjValLs   = zeros([numIterations, 1]);
vObjArg     = zeros([numIterations, 1]);

mAA = mA.' * mA;
vAb = mA.' * vB;

vX              = mA \ vB; %<! Initialization by the Least Squares Solution
vObjVal(1)      = hObjFun(vX);
vObjValLs(1)    = hObjFunLs(vX);
vObjArg(1)      = norm(vX - vXOpt);

for ii = 2:numIterations
    
    vG = (mAA * vX) - vAb + (2 * paramLambda * sign(vX) .* (abs(vX) - vA));
    vX = vX - (stepSize * vG);
    
    paramLambda = min(1.1 * paramLambda, paramLambdaMax);
    
    
    vObjVal(ii)     = hObjFun(vX);
    vObjValLs(ii)   = hObjFunLs(vX);
    vObjArg(ii)     = norm(vX - vXOpt);
end

disp([' ']);
disp(['Sub Gradient Descent Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(vObjVal(numIterations))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp([' ']);

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineSeries = plot(1:numIterations, [vObjVal, vObjValLs, optVal * ones([numIterations, 1])]);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(hLineSeries(3), 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Objective Function Value vs. Iteration']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Objective Function Value', ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numIterations]);
hLegend = ClickableLegend({['Sub Gradient Descent'], ['Sub Gradient Descent - LS'], ['Optimal Value (Brute Force)']});
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

