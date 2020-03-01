% StackExchange Signal Processing Q64035
% https://dsp.stackexchange.com/questions/64035
% Estimating the Signal by Deconvolution with a Prior on the Filter Coefficients
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     23/02/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

numCoefficients = 5;
numSamples      = 25;
noiseStd        = 0.5;

gridMinVal      = -2;
gridMaxVal      = 2;
numGridSamples  = 1001;

numIterations   = 100;
stopThr         = 1e-6;
convShape       = CONVOLUTION_SHAPE_SAME;
paramLambda     = (noiseStd * noiseStd) / numSamples;


%% Generate Data

vH = rand(numCoefficients, 1);
vH = vH / sum(vH);
vX = randn(numSamples, 1);
vX(vX < 0) = -1;
vX(vX >= 0) = 1;

vY = conv2(vX, vH, 'same');
vN = noiseStd * randn(size(vY, 1), 1);
vY = vY + vN;
% vY = CreateConvMtx1D(vH, numSamples, convShape) * vX;

vMu         = [-1; 1];
vSigma      = [0.05; 0.05];
vModelProb  = [0.5; 0.5]; %<! Sum must be 1

vG = linspace(gridMinVal, gridMaxVal, numGridSamples);
vG = vG(:);


%% Display Prior per Sample

vP = zeros(numGridSamples, 1);

for ii = 1:numGridSamples
    vP(ii) = CalcProbGmm(vMu, vSigma, vModelProb, vG(ii));
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosDefault); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hLineObj    = plot(vG, vP);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Prior Probability Density Function (PDF)']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['x']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['f(x)']}, ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Estimating the Signal and the Filter

hSumLogProb = @(vX) sum(log(max(CalcProbGmm(vMu, vSigma, vModelProb, vX), 1e-6)));

vXEst = sign(vY);
vHEst = zeros(numCoefficients, 1);

for ii = 1:numIterations
    % Estimate Filter Coefficients
    vHEst = EstimateFilterCoeff(vHEst, vXEst, vY, convShape, numIterations, stopThr);
    % Estimate Signal Samples
    vXEst = EstimateSignalSamples(vXEst, vHEst, vY, convShape, paramLambda, hSumLogProb);
end

vXEst = sign(vXEst);

all(sign(vY) == vXEst)
sum(sign(vY) == vX)
sum(vXEst == vX)


%% Display Results

hFigure     = figure('Position', [100, 100, 280, 320]); %<! [x, y, width, height]
hAxes       = axes('Units', 'pixels', 'Position', [17, 10, 256, 256]); %<! [x, y, width, height]
hImageObj   = imagesc(mA);
set(get(hAxes, 'Title'), 'String', {['Input Image A'], ['Objective Value - ', num2str(objValA)]}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(hAxes, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', []);
% set(hAxes, 'LooseInset', get(hAxes, 'TightInset'));
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

hFigure     = figure('Position', [100, 100, 280, 320]); %<! [x, y, width, height]
hAxes       = axes('Units', 'pixels', 'Position', [17, 10, 256, 256]); %<! [x, y, width, height]
hImageObj   = imagesc(mB);
set(get(hAxes, 'Title'), 'String', {['Input Image B'], ['Objective Value - ', num2str(objValB)]}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(hAxes, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', []);
% set(hAxes, 'LooseInset', get(hAxes, 'TightInset'));
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

