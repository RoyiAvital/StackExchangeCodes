% StackExchange Cross Validated Q401212
% https://stats.stackexchange.com/questions/401212
% Showing the Equivalence between the Regularized Regression Model and Constrained Regression Model Using KKT
%
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     12/04/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 2103;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;
verifyCvx       = OFF;


%% Simulation Parameters

numRows = 6;
numCols = 4;

paramTMinVal    = 0.1;
paramTMaxVal    = 200;
numSamples      = 2000;


%% Generate Data

mA              = randn(numRows, numCols);
vB              = randn(numRows, 1);
vParamT         = linspace(paramTMinVal, paramTMaxVal, numSamples);

mVX             = zeros(numCols, numSamples);
vParamLambda    = zeros(numSamples, 1);


%% Analysis

for ii = 1:numSamples
    
    paramT = vParamT(ii);
    [vX, paramLambda] = SolveLsNormSquaredConst(mA, vB, paramT, 25);
    
    vParamLambda(ii)    = paramLambda;
    mVX(:, ii)          = vX;
    
end


%% Solution by CVX

if(verifyCvx == ON)
    for ii = 1:numSamples
        cvxIdx = 50;
        
        cvx_begin('quiet')
        cvx_precision('best');
        variable vX(numCols)
        dual variable paramLambda;
        minimize( 0.5 * square_pos(  norm(mA * vX - vB, 2) ) );
        subject to
        paramLambda : square_pos(norm(vX, 2)) <= vParamT(ii);
        cvx_end
        
%         disp([' ']);
%         disp(['CVX Solution Summary']);
%         disp(['The CVX Solver Status - ', cvx_status]);
%         disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
%         disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);
%         disp([' ']);
        
        if(norm(mVX(:, ii) - vX(:)) > 1e-5)
            keyboard;
        end
        
        if(abs(paramLambda - vParamLambda(ii)) > 1e-5)
            keyboard;
        end
    end
    
end


%% Display Results

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
hLineObject = line(vParamT, vParamLambda);
set(hLineObject, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['The Mapping between \lambda and t']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 't', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', '\lambda', ...
    'FontSize', fontSizeAxis);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

