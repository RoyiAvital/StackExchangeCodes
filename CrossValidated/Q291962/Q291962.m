% Cross Validated Q291962
% https://stats.stackexchange.com/questions/291962
% Two Parameter Method of Moments Estimation
% References:
%   1.  Method of Moments - https://en.wikipedia.org/wiki/Method_of_moments_(statistics).
%   2.  Normal Distribution Moments - https://en.wikipedia.org/wiki/Normal_distribution#Moments.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     17/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numRows = 10;
numCols = 20;

paramEpsilon    = 0.1;


%% Generate Data

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);


%% Solution by CVX

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numCols)
    dual variable cvxLambda
    minimize( norm(vX, 1) );
    subject to
        cvxLambda : (0.5 * square_pos(norm((mA * vX) - vB, 2))) <= paramEpsilon;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Constratint Value Is Given By - [ ', num2str(sum(0.5 * sum((mA * vX - vB) .^ 2))), ' ]']);
disp([' ']);

sum(0.5 * sum((mA * vX - vB) .^ 2))


%% Solution by ADMM / PGM / CD

[vX, paramLambda] = SolveBp(mA, vB, paramEpsilon);

disp([' ']);
disp(['Basis Pursuit (ADMM) Solution Summary']);
disp(['The Optimal Value Is Given By - ', num2str(norm(vX, 1))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
disp(['The Constratint Value Is Given By - [ ', num2str(sum(0.5 * sum((mA * vX - vB) .^ 2))), ' ]']);
disp([' ']);


%% Analysis of Lambda & L1 Norm vs. Epsilon

epsMinVal       = 0.001;
epsMaxVal       = 2;
numSamples      = 100;
vParamEpsilon   = linspace(epsMinVal, epsMaxVal, numSamples).';

vParamLambda    = zeros([numSamples, 1]);
vL1Norm         = zeros([numSamples, 1]);

for ii = 1:numSamples
    paramEpsilon = vParamEpsilon(ii);
    [vX, paramLambda] = SolveBp(mA, vB, paramEpsilon);
    vParamLambda(ii) = paramLambda;
    vL1Norm(ii) = norm(vX);
end


%% Display Results

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineObject = line(vParamEpsilon, [vL1Norm, vParamLambda]);
set(hLineObject, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['The {L}_{1} Norm and \lambda as a Function of \epsilon']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', '\epsilon', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Value', ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend({['{L}_{1} Norm'], ['\lambda']});


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

