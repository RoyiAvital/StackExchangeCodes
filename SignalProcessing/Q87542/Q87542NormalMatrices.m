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

% Short
numSamples  = 5; %<! For x
numCoeff    = 3; %<! for h


%% Generate / Load Data

vX = reshape(1:numSamples, numSamples, 1);
vH = ones(numCoeff, 1);
vH = vH(:) / sum(vH);
vH = [1:numCoeff].'


%% Analysis

mHFull  = full(CreateConvMtx1D(vH, numSamples, CONVOLUTION_SHAPE_FULL));
mHSame  = full(CreateConvMtx1D(vH, numSamples, CONVOLUTION_SHAPE_SAME));
mHValid = full(CreateConvMtx1D(vH, numSamples, CONVOLUTION_SHAPE_VALID));


%% Display Results


hF = figure('Position', [100, 100, 800, 800]);

hA = subplot(1, 3, 1);
PlotMatrix(GenMhh(mHFull), 'setEqual', 0, 'titleString', 'Full Convolution');

hA = subplot(1, 3, 2);
PlotMatrix(GenMhh(mHSame), 'setEqual', 0, 'titleString', 'Same Convolution');

hA = subplot(1, 3, 3);
PlotMatrix(GenMhh(mHValid), 'setEqual', 0, 'titleString', 'Valid Convolution');

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

function [ mHH ] = GenMhh( mH )

mHH = mH.' * mH;

end






%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

