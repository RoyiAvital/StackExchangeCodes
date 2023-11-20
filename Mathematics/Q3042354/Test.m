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

subStreamNumberDefault = 5900;
subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 50;
numCols = numRows;

valA    = 0.23;
valB    = 1.05;

numIterations   = 5000;
stepSize        = 1 / 1000;


%% Generate Data

mA = randn(numRows, numCols);
mA = (mA.' * mA) + (0.1 * eye(numRows));
mA = mA + mA';

hObjFun = @(vX) 0.5 * vX.' * mA * vX;


%% Solution by CVX

cvx_begin('quiet')
    % cvx_precision('best');
    variable vX(numCols);
    minimize( 0.5 * quad_form(vX, mA) );
    subject to
        mA * vX >= valA;
        mA * vX <= valB;
cvx_end

vXRef  = vX;
optVal = cvx_optval;


%% Chambolle Pock Method

hGradP = @(vP, vX) mA * vP;
hProjP = @(vP) min(max(vP, valA), valB);
hStepX = @(vX, vP) -vP;
% hStepX = @(vX, vP) vX - stepSize * (mA * (vX + vP));

vX = mA \ (((valA + valB) / 2) * ones(numCols, 1));
vP = mA * vX;
mX = zeros(numCols, numIterations);
mX(:, 1) = vX;

% [vX, mX] = ChambollePock(vX, mX, vP, hGradP, hProjP, hStepX, stepSize, numIterations);

vObjValCp = zeros(numIterations, 1);
for ii = 1:numIterations
    vObjValCp(ii) = hObjFun(mX(:, ii));
end


%% Primal Dual Method
%   Solves 
%       min_x f(A * x) + g(x)

hProxF = @(vY, paramLambda) min(max(vY, valA), valB);
% Prox_{λFS}(y) = y - Prox_{F / λ}(y / λ)
hProxFS = @(vY, paramLambda) vY - paramLambda * hProxF(vY / paramLambda, 1 / paramLambda); %<! Prox of conjugate of F
hProxG = @(vY, paramLambda) (paramLambda * mA + eye(numRows)) \ vY;

% vP = mA \ (((valA + valB) / 2) * ones(numCols, 1));
% vX = mA * vP;
vP = mA \ (((valA + valB) / 2) * ones(numCols, 1));
vX = mA * vX;
mX = zeros(numCols, numIterations);
mX(:, 1) = vX;

paramTheta = 1;

[vX, mX] = PrimalDual(vX, mX, vP, mA, hProxFS, hProxG, paramTheta, numIterations);

vObjValPd = zeros(numIterations, 1);
for ii = 1:numIterations
    vObjValPd(ii) = hObjFun(mX(:, ii));
end


%% Display Reesults

save('Data.mat', 'mA', 'subStreamNumber', 'vXRef', 'optVal');

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
% hLineSeries = plot(1:numIterations, cvx_optval * ones(numIterations, 1), 'DisplayName', 'CVX');
% set(hLineSeries, 'LineWidth', lineWidthNormal);
% hLineSeries = plot(1:numIterations, vObjValCp, 'DisplayName', 'Chambolles Pock');
% set(hLineSeries, 'LineStyle', ':', 'LineWidth', lineWidthNormal);
hLineSeries = plot(1:numIterations, 20 * log10(abs(vObjValPd - cvx_optval) / abs(cvx_optval)), 'DisplayName', 'Primal Dual');
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Quadratic Form'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', ['$\frac{\left| F \left( {x}_{i} \right) - F \left( {x}^{\star} \right) \right|}{\left| F \left( {x}^{\star} \right) \right|}$ [dB]'], ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
hLegend = ClickableLegend();
set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

