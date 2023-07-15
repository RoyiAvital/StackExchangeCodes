% StackExchange Signal Processing Q22183
% https://dsp.stackexchange.com/questions/22183
% Primitive Feature Detection to Detect a Black Circle within an Image
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     15/07/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants



%% Parameters

% Data
img001FileName  = 'xvr6E.jpg'; %<! With black circle
img001Url       = 'https://i.stack.imgur.com/xvr6E.jpg';
img002FileName  = 'XVwSZ.jpg'; %<! Without black circle
img002Url       = 'https://i.stack.imgur.com/XVwSZ.jpg';

% Model
paramLambda = 400;


%% Generate / Load Data

if (~isfile(img001FileName))
    websave(img001FileName, img001Url);
end
if (~isfile(img002FileName))
    websave(img002FileName, img002Url);
end

mI = imread(img001FileName);
mI = im2double(mI);
mI = mean(mI, 3);

[numRows, numCols] = size(mI);


%% Analysis

% Smooth Image
mS = imbilatfilt(mI, 0.1, 5, "NeighborhoodSize", 21);
mT = repmat(mS, 1, 1, 3);

% Extract 2 profiles
mX = zeros(min(numRows, numCols), 2);
mX(:, 1) = diag(mS);
mX(:, 2) = diag(flip(mS, 1));

for ii = 1:min(numRows, numCols)
    mT(ii, ii, 1)   = 1;
    mT(ii, ii, 2:3) = 0;
end

mT = flip(mT, 2);
for ii = 1:min(numRows, numCols)
    mT(ii, ii, 1)   = 1;
    mT(ii, ii, 2:3) = 0;
end
mT = flip(mT, 2);



%% Display Results

figureIdx = figureIdx + 1;

vFigPos = [100, 100, numCols + 80, numRows + 80];

hF = figure('Position', vFigPos);
hA = axes(hF, 'Units', 'pixels', 'Position', [40, 40, numCols, numRows]);
set(hA, 'DataAspectRatio', [1, 1, 1]);
hImgObj = image(hA, repmat(mI, 1, 1, 3));
set(get(hA, 'Title'), 'String', {['Input Image']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


figureIdx = figureIdx + 1;

vFigPos = [100, 100, numCols + 80, numRows + 80];

hF = figure('Position', vFigPos);
hA = axes(hF, 'Units', 'pixels', 'Position', [40, 40, numCols, numRows]);
set(hA, 'DataAspectRatio', [1, 1, 1]);
hImgObj = image(hA, mT);
set(get(hA, 'Title'), 'String', {['Filtered Image and Profiles']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA   = axes(hF);
set(hA, 'NextPlot', 'add');
hLineObj = plot(mX);
for ii = 1:2
    set(hLineObj(ii), 'DisplayName', ['Profile ', num2str(ii)]);
end
set(get(hA, 'Title'), 'String', {['Profiles']}, ...
    'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Pixel Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Pixel Value']}, ...
    'FontSize', fontSizeAxis);

hLegend = ClickableLegend();

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

