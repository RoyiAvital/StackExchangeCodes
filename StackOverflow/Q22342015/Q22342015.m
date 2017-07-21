% Stack Overflow Q22342015
% https://stackoverflow.com/questions/22342015
% Best Way to Randomly Initialize Clusters in MATLAB
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     21/07/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

DISTANCE_METRIC_L1      = 1; %<! L1 Norm
DISTANCE_METRIC_L2      = 2; %<! L2 Norm
DISTANCE_METRIC_L_INF   = 3; %<! L Infinity Norm

INIT_METHOD_RANDOM              = 1;
INIT_METHOD_K_MEANS_PLUS_PLUS   = 2;

hDistMetricL1   = @(vX, vY) sum(abs(vX - vY));
hDistMetricL2   = @(vX, vY) sum((vX - vY) .^ 2);
hDistMetricLInf = @(vX, vY) max(abs(vX - vY));

numSamples      = 100; %<! Number of samples
dimSamples      = 2; %<! Dimensionality of each sample (Set to 2 for visualization)
numClusters     = 5; %<! Number of clusters
numIterations   = 15; %<! Number of iterations
distMetric      = DISTANCE_METRIC_L2;
initMethod      = INIT_METHOD_K_MEANS_PLUS_PLUS;


%% Generate Data and Samples

mX = rand([dimSamples, numSamples]);

% Setting Distance Function (Accespts 2 vectors and returns a scalar)
switch(distMetric)
    case(DISTANCE_METRIC_L1)
        hDistFun = hDistMetricL1;
    case(DISTANCE_METRIC_L2)
        hDistFun = hDistMetricL2;
    case(DISTANCE_METRIC_L_INF)
        hDistFun = hDistMetricLInf;
end


%% Results

switch(initMethod)
    case(INIT_METHOD_RANDOM)
        initMethodString = ['Random'];
        figureIdx = 1;
    case(INIT_METHOD_K_MEANS_PLUS_PLUS)
        initMethodString = ['K-Means++'];
        figureIdx = 2;
end

[vClusterId, mA] = InitKMeansClustering(mX, numClusters, initMethod);
[vClusterId, mA] = KMeansAlg(mX, mA, hDistFun, numIterations);

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
set(hAxes, 'NextPlot', 'add');
for jj = 1:numClusters
    
    scatterIdx = (2 * jj) - 1;
    
    hScatterObj(scatterIdx) = scatter(mX(1, vClusterId == jj), mX(2, vClusterId == jj));
    set(hScatterObj(scatterIdx), 'MarkerEdgeColor', mColorOrder(jj, :), 'LineWidth', lineWidthNormal, 'MarkerEdgeAlpha', 0.4);
    
    scatterIdx = 2 * jj;
    
    hScatterObj(scatterIdx) = scatter(mA(1, jj), mA(2, jj));
    set(hScatterObj(scatterIdx), 'MarkerEdgeColor', mColorOrder(jj, :), 'Marker', '+', 'SizeData', 256, 'LineWidth', lineWidthThick);
    
end
set(get(hAxes, 'Title'), 'String', {['K - Means, Initialization Method - ', initMethodString]}, 'Fontsize', fontSizeTitle);
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

