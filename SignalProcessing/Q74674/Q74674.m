% StackExchange Signal Processing Q74674
% https://dsp.stackexchange.com/questions/74674
% Simple Image Edge Preserving Filter
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     24/04/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

boxRadius       = 3;
numIterations   = 75;


%% Generate Data

mI = im2single(imread('../Q38542/Lena.png'));


%% Display the Signal and DFT

mO = ApplySideWindowFiltering(mI, boxRadius, numIterations);
% mO = SideWindowBoxFilter(mI, boxRadius, numIterations);

figure(); imshow(mI); figure(); imshow(mO);


figureIdx = figureIdx + 1;

hFigure = figure('Position', [100, 100, 532, 580]); %<! [x, y, width, height]
hAxes   = axes('Units', 'pixels', 'Position', [10, 10, 512, 512]); %<! [x, y, width, height]

hImgObj = image(repmat(mI, 1, 1, 3));
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['Input Image - Lena']}, ...
    'FontSize', fontSizeTitle);
set(hAxes, 'XTick', [], 'YTick', []);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(0, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

mO = mI;

for ii = 1:numIterations
    
    mO = ApplySideWindowFiltering(mO, boxRadius, 1);
    
    hImgObj = image(repmat(mO, 1, 1, 3));
    set(hAxes, 'DataAspectRatio', [1, 1, 1]);
    set(get(hAxes, 'Title'), 'String', {['Filtered Image - Lena'], ['Iteration Number - ', num2str(ii, '%03d')]}, ...
        'FontSize', fontSizeTitle);
    set(hAxes, 'XTick', [], 'YTick', []);
    
    pause(0.15);
    
    if(generateFigures == ON)
        % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
        print(hFigure, ['Figure', num2str(ii, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
    end
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

