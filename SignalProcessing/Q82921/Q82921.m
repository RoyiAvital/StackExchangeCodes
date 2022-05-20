% StackExchange Signal Processing Q82921
% https://dsp.stackexchange.com/questions/82921
% Slicing an Image into Tiles According to Content
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     20/05/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Simulation Constants

IMG_FILE_NAME = 'r0VEr.png';


%% Simulation Parameters

numAdjacentPixels = 2;


%% Generate / Load Data

[mT, mC] = imread(IMG_FILE_NAME); %<! Indexed image
mImg = ind2rgb(mT, mC); %<! Convert to RGB (Apply the colormap)


% mImg = im2double(imread(IMG_FILE_NAME));
mI = mean(im2double(mImg), 3);

numRows = size(mI, 1);
numCols = size(mI, 2);


%% Analysis

[vV, vM] = var(mI, 1, 1); %<! Variance and mean along the columns
vT = vV > eps();

vC = zeros(numCols, 1); %<! Class per value
currClass = 1;
vC(1) = currClass;
numDividers = 1; %<! Ay least one diver at most left

% Segmenting the data
for ii = 2:numCols
    if(vT(ii) ~= vT(ii - 1))
        currClass = currClass + 1;
        if(~vT(ii))
            numDividers = numDividers + 1;
        end
    end
    vC(ii) = currClass;
end

numClasses = currClass;

mSupport = zeros(numClasses, 3); %<! Start Idx, End Idx, Value
mSupport(1, 1) = 1;
mSupport(1, 3) = vT(1);

classIdx = 1;
for ii = 2:numCols
    if(vC(ii) ~= vC(ii - 1))
        mSupport(classIdx, 2) = ii - 1;
        classIdx = classIdx + 1;
        mSupport(classIdx, 1) = ii;
        mSupport(classIdx, 3) = vT(ii);
    end
end
mSupport(classIdx, 2) = numCols;

vDividers = zeros(numDividers, 1); %<! The column index per divider

vDividers(:) = round(mean(mSupport(mSupport(:, 3) == 0, 1:2), 2));

% Find the center index per segment of zeros


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', [100, 100, 569, 766], 'Units', 'pixels');
hAxes = axes(hFigure, 'Units', 'pixels', 'Position', [30, 520, 509, 206]);
% hAxes   = axes(hFigure, 'Position', [20, 520, 509, 206], 'Units', 'pixels');
hImgObj = image(mImg);
set(get(hAxes, 'Title'), 'String', {['Icons Image']}, ...
    'FontSize', fontSizeTitle);
for ii = 1:numDividers
    xline(hAxes, vDividers(ii), 'r', 'LineWidth', lineWidthNormal);
end

% hLegend = ClickableLegend({[''], ['Dividers']});


hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [30, 280, 509, 200]);
hLineObj = plot(1:numCols, vT);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Thresholed Variance per Column']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Column Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numCols]);
for ii = 1:numDividers
    xline(hAxes, vDividers(ii), 'r', 'LineWidth', lineWidthNormal);
end


hAxes   = axes(hFigure, 'Units', 'pixels', 'Position', [30, 40, 509, 200]);
hLineObj = scatter(1:numCols, vT, markerSizeMedium, vC);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Segments of the Variance per Column']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Column Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value']}, ...
    'FontSize', fontSizeAxis);
set(hAxes, 'XLim', [1, numCols]);
for ii = 1:numDividers
    xline(hAxes, vDividers(ii), 'r', 'LineWidth', lineWidthNormal);
end

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

