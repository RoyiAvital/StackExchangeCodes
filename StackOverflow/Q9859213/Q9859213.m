% Stack Overflow Q9859213
% https://stackoverflow.com/questions/9859213
% K-Means Algorithm with Arbitrary Distance Function Matlab (Chebyshev Distance)
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     20/07/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

DISTANCE_TYPE_EUCLIDEAN = 1; %<! L2 Norm
DISTANCE_TYPE_MANHATTAN = 2; %<! L1 Norm
DISTANCE_TYPE_CHEBYSHEV = 3; %<! L Infinity Norm

hEuclideanDist = @(vX, vY) sum((vX - vY) .^ 2);
hManhattanDist = @(vX, vY) sum(abs(vX - vY));
hChebyshevDist = @(vX, vY) max(abs(vX - vY));

numSamples  = 100;
dimSamples  = 2; %<! For Visualization
numClusters = 5; %<! Number of Clusters
numIter     = 15; %<! Number of Iterations
distType    = DISTANCE_TYPE_MANHATTAN;


%% Generate Data and Samples

mX = rand([dimSamples, numSamples]);


% Setting Distance Function (Accespts 2 vectors and returns a scalar)
switch(distType)
    case(DISTANCE_TYPE_EUCLIDEAN)
        hDistFun = hEuclideanDist;
        distString = ['Euclidean'];
    case(DISTANCE_TYPE_MANHATTAN)
        hDistFun = hManhattanDist;
        distString = ['Manhattan'];
    case(DISTANCE_TYPE_CHEBYSHEV)
        hDistFun = hChebyshevDist;
        distString = ['Chebyshev'];
end


%% Results

tic();
mA          = mX(:, randperm(numSamples, numClusters)); %<! Cluster Centroids

hFigure = figure('Position', figPosLarge);
hAxes   = axes();

for ii = 1:numIter
    
    for kk = 1:numSamples
        vX = mX(:, kk);
        
        minDist = inf;
        for ll = 1:numClusters
            vY = mA(:, ll);
            currDist = hDistFun(vX, vY);
            
            if(currDist < minDist)
                minDist         = currDist;
                vClusterId(kk)  = ll;
            end
        end
        
    end
    
    for jj = 1:numClusters
        mA(:, jj) = sum(mX(:, vClusterId == jj), 2) ./ sum(vClusterId == jj);
    end
    
    set(hAxes, 'NextPlot', 'replace');
    
    set(hAxes, 'NextPlot', 'add');
    for jj = 1:numClusters
        
        scatterIdx = (2 * jj) - 1;
        
        hScatterObj(scatterIdx) = scatter(mX(1, vClusterId == jj), mX(2, vClusterId == jj));
        set(hScatterObj(scatterIdx), 'MarkerEdgeColor', mColorOrder(jj, :), 'LineWidth', lineWidthNormal, 'MarkerEdgeAlpha', 0.4);
        
        scatterIdx = 2 * jj;
        
        hScatterObj(scatterIdx) = scatter(mA(1, jj), mA(2, jj));
        set(hScatterObj(scatterIdx), 'MarkerEdgeColor', mColorOrder(jj, :), 'Marker', '+', 'SizeData', 256, 'LineWidth', lineWidthThick);
        
    end
    set(get(hAxes, 'Title'), 'String', {['K - Means - Iteration #', num2str(ii, '%03d')], ['Distance Type - ', distString]}, ...
        'Fontsize', fontSizeTitle);
    
    set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
    
    drawnow();
    pause(0.2);
    
    if(generateFigures == ON)
        saveas(hFigure,['Figure', num2str(ii, figureCounterSpec), '.png']);
    end
    
    if(ii ~= numIter)
        set(hScatterObj, 'XData', [], 'YData', []);
    end
    
    
end
runTime = toc();


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

