% StackExchange Signal Processing Q76344
% https://dsp.stackexchange.com/questions/76344
% Generate the Matrix Form of 1D Convolution Kernel
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     19/07/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

vK              = [1; 4; 6; 4; 1] / 16; %<! Regularization
numSamples      = 50;
vConvShape      = [1; 2; 3];
cConvShapeStr   = {['Full'], ['Same'], ['Valid']};


%% Generate Data

cConvMtx = cell(3, 1);

for ii = 1:length(vConvShape)
    cConvMtx{ii} = CreateConvMtx1D(vK, numSamples, vConvShape(ii));
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure     = figure('Position', [100, 100, 1200, 600]);
for ii = 1:length(vConvShape)
    hAxes       = subplot(1, 3, ii);
    hScatterObj = ScatterSparse(cConvMtx{ii}, 'fill');
    set(hAxes, 'YDir', 'reverse');
    set(hAxes, 'DataAspectRatio', [1, 1, 1]);
    set(hAxes, 'YLim', [0, numSamples + size(vK, 1) - 1]);
    % set(hLineObj, 'LineWidth', lineWidthNormal);
    set(get(hAxes, 'Title'), 'String', {['Convolution Type: ', cConvShapeStr{ii}]}, ...
        'FontSize', fontSizeTitle);
    % set(get(hAxes, 'XLabel'), 'String', {['Sample Index']}, ...
    %     'FontSize', fontSizeAxis);
    % set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    %     'FontSize', fontSizeAxis);
    % hLegend = ClickableLegend({['Ground Truth Signal'], ['Measured Signal']});
end

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

