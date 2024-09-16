% StackExchange Signal Processing Q95071
% https://dsp.stackexchange.com/questions/95071
% Build the Laplacian Matrix of Edge Preserving Multiscale Image Decomposition based on Local Extrema
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     16/09/2024
%   *   First release.


%% Settings

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

% Image from the paper (Cropped and resized)
imgUrl = 'https://i.postimg.cc/85Jjs9wJ/Flowers.png'; %<! https://i.imgur.com/PckT6jF.png

paramK = 5; %<! Radius of the local extrema
paramN = 1; %<! Radius of the local Laplacian

% 4 Point connectivity, excludes center for no cyclic graph
hV = @(ii, jj, mm, nn) (mm * nn == 0) & ((mm ~= 0) || (nn ~= 0)); %<! ii, jj ref pixel location, mm, nn neighbor pixel shift
hW = @(valI, valN) abs(valI - valN); %<! Reference value vs. neighbor (r <-> s in the paper)

epsVal = 1e-5;


%% Generate / Load Data

mI = im2double(imread(imgUrl));
mI = mean(mI, 3); %<! RGB Image
mI = imresize(mI, [256, 256]);
mI = max(mI, 0);

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx = numRows * numCols;


%% Analysis

vPxInd = 1:numPx;

localLen = (2 * paramK) + 1; %<! k in the paper

% Local Extrema
mLocalKValue = ordfilt2(mI, localLen * localLen - localLen + 1, ones(localLen), 'symmetric');
mLocalMax = mI >= mLocalKValue; %<! Local Maximum
mLocalKValue = ordfilt2(mI, localLen, ones(localLen), 'symmetric');
mLocalMin = mI <= mLocalKValue; %<! Local Minimum

% Graph
% Local variance image
mV = stdfilt(mI, true(2 * paramN + 1));
mV = mV .* mV;

mW = BuildGraphMatrix(mI, hV, hW, paramN);
% Scaling
% minVal = min(nnz(mW));
% maxVal = max(nnz(mW));
% mW = (mW - minVal) / (maxVal - minVal);
% Apply the Gaussian Function
% mW = mW ./ (2 * mV(:)); %<! Wont work! Converts to full

[vR, vC, vVals] = find(mW);
for ii = 1:numPx
    vRowIndx = vR == ii;
    localVar = max(mV(ii), 0.0001);
    vVals(vRowIndx) = exp(vVals(vRowIndx) / (2 * localVar)); %<! Exponent function
    vVals(vRowIndx) = vVals(vRowIndx) / sum(vVals(vRowIndx)); %<! Unit Sum
end
mW = sparse(vR, vC, vVals, numPx, numPx);

% Graph Laplacian
mD = diag(sum(mW, 2)); %<! Degree Matrix
mL = mD - mW; %<! Laplacian Matrix

% Interpolate Maximum
vV = find(mLocalMax);
vU = setdiff(vPxInd(:), vV);

mLu = mL(vU, vU); %<! The Laplacian sub matrix to optimize by
mR  = mL(vU, vV);
oLu = decomposition(mLu);

vXv  = mI(vV); %<! Anchor values
vXu = -(oLu \ (mR * vXv));






%% Display Results


% hF = figure('Position', [100, 100, 1400, 500]);
% 
% hA = subplot(1, 3, 1);
% set(hA, 'NextPlot', 'add');
% hLineObj = plot(hA, vXFull, 'DisplayName', 'Direct Matrix Solver (Ref)');
% set(hLineObj, 'LineStyle', 'none');
% set(hLineObj, 'Marker', 'o');
% hLineObj = plot(hA, vXXFull, 'DisplayName', 'Iterative Solver (Convolution)');
% set(hLineObj, 'LineStyle', 'none');
% set(hLineObj, 'Marker', '+');
% set(get(hA, 'Title'), 'String', {['Solution for "full" Convolution']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hA, 'YLabel'), 'String', {['Sample Value']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend();
% 
% hA = subplot(1, 3, 2);
% set(hA, 'NextPlot', 'add');
% hLineObj = plot(hA, vXSame, 'DisplayName', 'Direct Matrix Solver (Ref)');
% set(hLineObj, 'LineStyle', 'none');
% set(hLineObj, 'Marker', 'o');
% hLineObj = plot(hA, vXXSame, 'DisplayName', 'Iterative Solver (Convolution)');
% set(hLineObj, 'LineStyle', 'none');
% set(hLineObj, 'Marker', '+');
% set(get(hA, 'Title'), 'String', {['Solution for "same" Convolution']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hA, 'YLabel'), 'String', {['Sample Value']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend();
% 
% hA = subplot(1, 3, 3);
% set(hA, 'NextPlot', 'add');
% hLineObj = plot(hA, vXValid, 'DisplayName', 'Direct Matrix Solver (Ref)');
% set(hLineObj, 'LineStyle', 'none');
% set(hLineObj, 'Marker', 'o');
% hLineObj = plot(hA, vXXValid, 'DisplayName', 'Iterative Solver (Convolution)');
% set(hLineObj, 'LineStyle', 'none');
% set(hLineObj, 'Marker', '+');
% set(get(hA, 'Title'), 'String', {['Solution for "valid" Convolution']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hA, 'YLabel'), 'String', {['Sample Value']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend();
% 
% if(generateFigures == ON)
%     % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end

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

