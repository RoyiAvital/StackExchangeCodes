% StackExchange Signal Processing Q52760
% https://dsp.stackexchange.com/questions/52760
% Convolution Strategy / Method for the Fastest 1D Convolution
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     27/04/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_METHOD_DIRECT       = 1;
CONVOLUTION_METHOD_DFT          = 2;
CONVOLUTION_METHOD_OVERLAP_SAVE = 3;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;

vConvMethod = [CONVOLUTION_METHOD_DIRECT, CONVOLUTION_METHOD_DFT, CONVOLUTION_METHOD_OVERLAP_SAVE];
vConvShape  = [CONVOLUTION_SHAPE_FULL, CONVOLUTION_SHAPE_SAME, CONVOLUTION_SHAPE_VALID];


%% Simulation Parameters

vNumSamples = [4:32:2048];
convShape   = CONVOLUTION_SHAPE_SAME;


%% Generate / Load Data

numSamplesSteps = length(vNumSamples);
numMethods      = length(vConvMethod);

vS = randn(vNumSamples(numSamplesSteps), 1);
vK = randn(vNumSamples(numSamplesSteps), 1);


%% Run Time Analysis

switch(convShape)
    case(CONVOLUTION_SHAPE_FULL)
        convString = 'full';
    case(CONVOLUTION_SHAPE_SAME)
        convString = 'same';
    case(CONVOLUTION_SHAPE_VALID)
        convString = 'valid';
end

mRunTime = zeros(numSamplesSteps, numSamplesSteps, numMethods);


for jj = 1:numSamplesSteps
    % Kernel Length
    numSamplesKernel = vNumSamples(jj);
    vKK = vK(1:numSamplesKernel);
    for ii = jj:numSamplesSteps
        % Signal Length
        numSamplesSignal = vNumSamples(ii);
        vSS = vS(1:numSamplesSignal);
        for kk = 1:numMethods
            switch(vConvMethod(kk))
                case(CONVOLUTION_METHOD_DIRECT)
                    hF = @() conv2(vSS, vKK, convString);
                case(CONVOLUTION_METHOD_OVERLAP_SAVE)
                    hF = @() ConvolutionOverlapSave(vSS, vKK, convShape);
                case(CONVOLUTION_METHOD_DFT)
                    hF = @() ConvolutionDft(vSS, vKK, convShape);
            end
            
            mRunTime(ii, jj, kk) = timeit(hF);
            
        end
    end
end

[~, mBestMethod] = min(mRunTime, [], 3);
for jj = 1:numSamplesSteps
    for ii = 1:numSamplesSteps
        if(jj > ii)
            mBestMethod(ii, jj) = 0;
        end
    end
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge); %<! [x, y, width, height]
hAxes       = axes(); %<! [x, y, width, height]
hImageObj   = imagesc(vNumSamples, vNumSamples, mBestMethod);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['Linear Convolution Run Time Comparison']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Kernel Length']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Signal Length']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

