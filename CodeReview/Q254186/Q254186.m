% StackExchange Code Review Q254186
% https://codereview.stackexchange.com/questions/254186
% Calculation of the Distance Matrix in the K-Means Algorithm in MATLAB
% References:
%   1.  
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     01/01/2021  Royi Avital     RoyiAvital@yahoo.com
%   *   First release.


%% General Parameters

subStreamNumberDefault = 122; %<! Set to 0 for Random

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

userDataFileName = 'UserData.mat';

numIterations   = 10000;
numClusters     = 3;
hDistFun        = @(mA, mB) CalcDistanceMatrixCols(mA, mB);  %<! The fastest implementation
stopTol         = 1e-6;
dispFig         = ON;
saveFig         = ON;
dispTime        = 2; %<! [Sec]


%% Load / Generate Data

load(userDataFileName);
mXT = mX.';

numSamples = size(mX, 1); %<! Each sample is a row

hDistFun1C = @() CalcDistanceMatrixACols(mXT); %<! User Data is Rows
hDistFun1R = @() CalcDistanceMatrixARows(mX);
hDistFun2C = @() CalcDistanceMatrixBCols(mXT); %<! User Data is Rows
hDistFun2R = @() CalcDistanceMatrixBRows(mX);
hDistFun3C = @() CalcDistanceMatrixCCols(mXT); %<! User Data is Rows
hDistFun3R = @() CalcDistanceMatrixCRows(mX);

hDistFunC = @() CalcDistanceMatrixCols(mXT, mXT); %<! User Data is Rows
hDistFunR = @() CalcDistanceMatrixRows(mX, mX);


%% Run Time Single Data

cDistFun = {hDistFun1C, hDistFun1R, hDistFun2C, hDistFun2R, hDistFun3C, hDistFun3R};
cDistFunString = {['Dist Fun 1 Columns'], ['Dist Fun 1 Rows'], ['Dist Fun 2 Columns'], ...
    ['Dist Fun 2 Rows'], ['Dist Fun 3 Columns'], ['Dist Fun 3 Rows']};

numDistFun = length(cDistFun);

vRunTime = zeros(numDistFun, 1);

for ii = 1:numDistFun
    vRunTime(ii) = timeit(cDistFun{ii}, 1);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hBarObj     = bar(categorical(cDistFunString), 1e6 * vRunTime);
% set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Distance Function Single Input'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Function', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Run Time [Micro Seconds]', ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);
% hLegend = ClickableLegend(cMethodString);
% set(hLegend, 'FontSize', fontSizeAxis);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Run Time Double Data

cDistFun = {hDistFunC, hDistFunR};
cDistFunString = {['Dist Fun Double Data Columns'], ['Dist Fun Double Data Rows']};

numDistFun = length(cDistFun);

vRunTime = zeros(numDistFun, 1);

for ii = 1:numDistFun
    vRunTime(ii) = timeit(cDistFun{ii}, 1);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hBarObj     = bar(categorical(cDistFunString), 1e6 * vRunTime);
% set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['Distance Function Double Input'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Function', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Run Time [Micro Seconds]', ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);
% hLegend = ClickableLegend(cMethodString);
% set(hLegend, 'FontSize', fontSizeAxis);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Running K-Means

% Working on data where each smaple is the a column vector as it is faster

mC = mXT(:, randperm(numSamples, numClusters));

[vClusterIdx, mC, vCostFun] = ClusterKMeans(mXT, mC, hDistFun, numIterations, stopTol, dispFig, saveFig, dispTime);

if(any(vCostFun == -1))
    numIterations = find(vCostFun == -1, 1, 'first') - 1;
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numIterations, vCostFun(1:numIterations));
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', ['K-Means Objective Function'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Iteration Number', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Function Value', ...
    'FontSize', fontSizeAxis);
% set(hAxes, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);
% hLegend = ClickableLegend(cMethodString);
% set(hLegend, 'FontSize', fontSizeAxis);

if(generateFigures == ON)
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

