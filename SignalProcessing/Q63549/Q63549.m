% StackExchange Signal Processing Q63549
% https://dsp.stackexchange.com/questions/63549
% Computationally Efficient Ways of Determining If Pixels Are Clumped
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     06/01/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageFileName001 = 'Image001.png';
imageFileName002 = 'Image002.png';


%% Generate Data

mA = imread(imageFileName001);
mB = imread(imageFileName002);

mA = ~logical(mA(:, :, 1));
mB = ~logical(mB(:, :, 1));

% Removing outer frame
mA(1, :) = false;
mA(end, :) = false;
mA(:, 1) = false;
mA(:, end) = false;

mB(1, :) = false;
mB(end, :) = false;
mB(:, 1) = false;
mB(:, end) = false;

% Extracting the properties of the Connected Components
% Can use half of 'MajorAxisLength' for the Radius.
% Then from 'PixelIdxList' do the computation
%{
objValA = CalcImageObjValImgBinaryImageProps(mA);
objValB = CalcImageObjValImgBinaryImageProps(mB);
%}


% Doing the Same, just with a nice trick to get the circle parameters by
% Binray Image Distnace Transform of the inverted image.
objValA = CalcImageObjValImgBinaryImageDistanceTransform(mA);
objValB = CalcImageObjValImgBinaryImageDistanceTransform(mB);


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

