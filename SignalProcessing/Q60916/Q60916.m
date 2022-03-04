% StackExchange Signal Processing Q60916
% https://dsp.stackexchange.com/questions/60916
% What Is the Bilateral Filter Category: LPF, HPF, BPF or BSF?
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.001     02/03/2022
%   *   Using 'sgtitle()' instead of 'suptitle()' (Requires R2018b and above).
% - 1.0.000     19/08/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

inputImageFileName  = 'Lenna.png';

kernelRadius    = 7;
spatialStd      = 2;
vRangeStd       = [0.001, 0.01, 0.05, 0.5, 1e10];


%% Generate Data

mI = im2double(imread(inputImageFileName));
mI = mean(mI, 3);
mO = repmat(mI, [1, 1, 3]);

mRefPixels  = [84, 432; 82, 442; 267, 159; 420, 320];
mColors     = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 0, 1];


%% Analysis

numPatches      = size(mRefPixels, 1);
kernelLength    = (2 * kernelRadius) + 1;

for ii = 1:numPatches
    mO(mRefPixels(ii, 1), mRefPixels(ii, 2), :) = reshape(mColors(ii, :), [1, 1, 3]);
    mO(mRefPixels(ii, 1) - kernelRadius - 1, mRefPixels(ii, 2) - kernelRadius - 1:mRefPixels(ii, 2) + kernelRadius + 1, :) = repmat(reshape(mColors(ii, :), [1, 1, 3]), [1, kernelLength + 2, 1]);
    mO(mRefPixels(ii, 1) + kernelRadius + 1, mRefPixels(ii, 2) - kernelRadius - 1:mRefPixels(ii, 2) + kernelRadius + 1, :) = repmat(reshape(mColors(ii, :), [1, 1, 3]), [1, kernelLength + 2, 1]);
    mO(mRefPixels(ii, 1) - kernelRadius - 1:mRefPixels(ii, 1) + kernelRadius + 1, mRefPixels(ii, 2) - kernelRadius - 1, :) = repmat(reshape(mColors(ii, :), [1, 1, 3]), [kernelLength + 2, 1, 1]);
    mO(mRefPixels(ii, 1) - kernelRadius - 1:mRefPixels(ii, 1) + kernelRadius + 1, mRefPixels(ii, 2) + kernelRadius + 1, :) = repmat(reshape(mColors(ii, :), [1, 1, 3]), [kernelLength + 2, 1, 1]);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes;
hImageObj   = imshow(mO);
set(get(hAxes, 'Title'), 'String', {['Lenna Image with Patches']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


for ii = 1:length(vRangeStd)
    rangeStd = vRangeStd(ii);
    
    figureIdx = figureIdx + 1;
    hFigure = figure('Position', figPosLarge);
    
    plotIdx = 0;
    patchIdx = 0;
    
    for jj = 1:4
        patchIdx = patchIdx + 1;
        plotIdx = plotIdx + 1;
        
        hAxes       = subplot(4, 5, plotIdx);
        hImageObj   = imshow(mI(mRefPixels(patchIdx, 1) - kernelRadius:mRefPixels(patchIdx, 1) + kernelRadius, mRefPixels(patchIdx, 2) - kernelRadius:mRefPixels(patchIdx, 2) + kernelRadius));
        set(get(hAxes, 'Title'), 'String', {['Input Patch']}, ...
            'FontSize', fontSizeTitle);
        
        plotIdx = plotIdx + 1;
        
        % Spatial Weight
        mWs = CalcSpatialWeights(kernelRadius, spatialStd);
        
        hAxes       = subplot(4, 5, plotIdx);
        hImageObj   = imshow(mWs, []);
        set(get(hAxes, 'Title'), 'String', {['Spatial Weight']}, ...
            'FontSize', fontSizeTitle);
        
        plotIdx = plotIdx + 1;
        
        % Spatial Weight
        mWr = CalcRangeWeights(mI, mRefPixels(patchIdx, :), kernelRadius, rangeStd);
        
        hAxes       = subplot(4, 5, plotIdx);
        hImageObj   = imshow(mWr, []);
        set(get(hAxes, 'Title'), 'String', {['Range Weight']}, ...
            'FontSize', fontSizeTitle);
        
        plotIdx = plotIdx + 1;
        
        % Kernel Weight
        mW = mWr .* mWs;
        mW = mW / sum(mW(:));
        
        hAxes       = subplot(4, 5, plotIdx);
        hImageObj   = imshow(mW, []);
        set(get(hAxes, 'Title'), 'String', {['Kernel Weight']}, ...
            'FontSize', fontSizeTitle);
        
        plotIdx = plotIdx + 1;
        
        hAxes       = subplot(4, 5, plotIdx);
        hImageObj   = imshow(log(1 + abs(fftshift(fft2(mW)))), []);
        set(get(hAxes, 'Title'), 'String', {['Kernel - Frequency']}, ...
            'FontSize', fontSizeTitle);
        
    end

    sgtitle(hFigure, ['Kernel Analysis for rangeStd - ', num2str(rangeStd)]);
    
    if(generateFigures == ON)
        saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    end
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

