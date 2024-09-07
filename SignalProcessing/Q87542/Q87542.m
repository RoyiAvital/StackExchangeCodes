% StackExchange Signal Processing Q87542
% https://dsp.stackexchange.com/questions/87542
% Solving Linear Equation of Discrete Convolution Kernels
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     06/04/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Parameters

numSamples  = 10; %<! For x
numCoeffH   = 3; %<! for h
numCoeffG   = 2; %<! for g

paramLambda = 0.5;


%% Generate / Load Data

vY = rand(numSamples, 1);
vH = rand(numCoeffH, 1);
vG = rand(numCoeffG, 1);


%% Analysis

mHFull  = full(CreateConvMtx1D(vH, numSamples, CONVOLUTION_SHAPE_FULL));
mHSame  = full(CreateConvMtx1D(vH, numSamples, CONVOLUTION_SHAPE_SAME));
mHValid = full(CreateConvMtx1D(vH, numSamples, CONVOLUTION_SHAPE_VALID));

mGFull  = full(CreateConvMtx1D(vG, numSamples, CONVOLUTION_SHAPE_FULL));
mGSame  = full(CreateConvMtx1D(vG, numSamples, CONVOLUTION_SHAPE_SAME));
mGValid = full(CreateConvMtx1D(vG, numSamples, CONVOLUTION_SHAPE_VALID));

mAFull  = BuildA(mHFull, mGFull, paramLambda);
mASame  = BuildA(mHSame, mGSame, paramLambda);
mAValid = BuildA(mHValid, mGValid, paramLambda);

% Direct Method as a Reference

% sLinSolv.SYM    = true();
% sLinSolv.POSDEF = true();
% vXFull = linsolve(mAFull, vY, sLinSolv);
vXFull  = mAFull \ vY;
vXSame  = mASame \ vY;
vXValid = mAValid \ vY;

% Conjugate Gradient Using
hNormEqnFull    = @(vX) ConvNormalEquationsFull(vX, vH, vG, paramLambda);
hNormEqnSame    = @(vX) ConvNormalEquations(vX, vH, vG, CONVOLUTION_SHAPE_SAME, paramLambda);
hNormEqnValid   = @(vX) ConvNormalEquations(vX, vH, vG, CONVOLUTION_SHAPE_VALID, paramLambda);

vXXFull     = pcg(hNormEqnFull, vY);
vXXSame     = pcg(hNormEqnSame, vY);
vXXValid    = pcg(hNormEqnValid, vY);



%% Display Results


hF = figure('Position', [100, 100, 1400, 500]);

hA = subplot(1, 3, 1);
set(hA, 'NextPlot', 'add');
hLineObj = plot(hA, vXFull, 'DisplayName', 'Direct Matrix Solver (Ref)');
set(hLineObj, 'LineStyle', 'none');
set(hLineObj, 'Marker', 'o');
hLineObj = plot(hA, vXXFull, 'DisplayName', 'Iterative Solver (Convolution)');
set(hLineObj, 'LineStyle', 'none');
set(hLineObj, 'Marker', '+');
set(get(hA, 'Title'), 'String', {['Solution for "full" Convolution']}, ...
    'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend();

hA = subplot(1, 3, 2);
set(hA, 'NextPlot', 'add');
hLineObj = plot(hA, vXSame, 'DisplayName', 'Direct Matrix Solver (Ref)');
set(hLineObj, 'LineStyle', 'none');
set(hLineObj, 'Marker', 'o');
hLineObj = plot(hA, vXXSame, 'DisplayName', 'Iterative Solver (Convolution)');
set(hLineObj, 'LineStyle', 'none');
set(hLineObj, 'Marker', '+');
set(get(hA, 'Title'), 'String', {['Solution for "same" Convolution']}, ...
    'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend();

hA = subplot(1, 3, 3);
set(hA, 'NextPlot', 'add');
hLineObj = plot(hA, vXValid, 'DisplayName', 'Direct Matrix Solver (Ref)');
set(hLineObj, 'LineStyle', 'none');
set(hLineObj, 'Marker', 'o');
hLineObj = plot(hA, vXXValid, 'DisplayName', 'Iterative Solver (Convolution)');
set(hLineObj, 'LineStyle', 'none');
set(hLineObj, 'Marker', '+');
set(get(hA, 'Title'), 'String', {['Solution for "valid" Convolution']}, ...
    'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

% figureIdx = figureIdx + 1;
% 
% hF = figure('Position', figPosLarge);
% hA   = axes(hF);
% set(hA, 'NextPlot', 'add');
% hLineObj = plot(-mU1(:, 1), 'DisplayName', 'Kernel A');
% set(hLineObj, 'LineWidth', lineWidthNormal);
% hLineObj = plot(-mU2(:, 1), 'DisplayName', 'Kernel B');
% set(hLineObj, 'LineWidth', lineWidthNormal);
% set(get(hA, 'Title'), 'String', {['Separable Filters of the Kernels']}, ...
%     'FontSize', fontSizeTitle);
% hLegend = ClickableLegend();
% 
% if(generateFigures == ON)
%     % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end



%% Auxiliary Functions

function [ mA ] = BuildA( mH, mG, paramLambda )

mA = ((mH.' * mH) + (paramLambda * (mG.' * mG)));


end






%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

