% StackExchange Signal Processing Q71660
% https://dsp.stackexchange.com/questions/71660
% Deconvolution with Noisy Measurement of the Model Coefficients
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     01/04/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 4464;
subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Parameters

numGridPts = 250;

vG = linspace(0, 5, numGridPts);
vG = vG(:);

modelOrder = 2;

noiseStd = 0.25;


%% Generate / Load Data

vX = rand(modelOrder + 1, 1);
mH = vG .^ [0:modelOrder];
vY = mH * vX;

mHH = (vG + (noiseStd * randn(size(vG)))) .^ [0:modelOrder];
mHH = mH + (noiseStd * randn(size(mH)));
vYY = vY + (noiseStd * randn(size(vY)));


%% Analysis

vXLS    = mHH \ vYY; %<! Least Squares Solution
vXTLS   = TlsRegression(mHH, vYY); %<! Total Least squares Solution
% vXTLS2  = TlsRegression2(mHH, vYY); %<! Total Least squares Solution
vXTLS3 = TlsRegression(mHH - mean(mHH, 1), vY);

mseLS  = mean((vXLS - vX) .^ 2);
mseTLS = mean((vXTLS - vX) .^ 2);

% mseTLS2 = mean((vXTLS2 - vX) .^ 2);


%% Display Results

disp(['The Least Squares solution MSE vs. the ground truth      : ', num2str(mseLS)]);
disp(['The Total Least Squares solution MSE vs. the ground truth: ', num2str(mseTLS)]);


figureIdx = figureIdx + 1;

hF = figure('Position', figPosXLarge);
hA = axes(hF);
set(hA, 'NextPlot', 'add');
hScatterObj = scatter(vG, vY, 'fill', 'DisplayName', 'Ground Truth');
% set(hLineObj, 'LineWidth', lineWidthNormal);
hScatterObj = scatter(mHH(:, 2), vYY, 'fill', 'DisplayName', 'Measurements');
hLineObj = plot(vG, mH * vXLS, 'DisplayName', 'Least Squares');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(vG, mH * vXTLS, 'DisplayName', 'Total Least Squares');
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hA, 'Title'), 'String', {['Least squares vs. Total Least Squares']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions

function [ vX ] = TlsRegression( mH, vY )

numCols = size(mH, 2);

[~, ~, mV] = svd([mH, vY]);
vX = -mV(1:numCols, numCols + 1) / mV(numCols + 1, numCols + 1);
% For analysis
[~, vD, ~] = svd([mH, vY], 'vector');
disp(vD(end));

end

function [ vX ] = TlsRegression2( mH, vY )

numCols = size(mH, 2);

% [~, ~, mV] = svd([mH, vY]);
% vX = -mV(1:numCols, numCols + 1) / mV(numCols + 1, numCols + 1);
[~, vD, ~] = svd([mH, vY], 'vector');
disp(vD(end));

vX = ((mH.' * mH) - ((vD(end) * vD(end)) * eye(numCols))) \ (mH.' * vY);

end

function [ vX ] = TlsRegression3( mH, vY )

numCols = size(mH, 2);
mHH = [mH, vY];
mHH = mHH - mean(mHH, 1);

[~, ~, mV] = svd(mHH);
vX = -mV(1:numCols, numCols + 1) / mV(numCols + 1, numCols + 1);

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

