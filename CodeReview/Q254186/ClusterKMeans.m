function [ vClusterIdx, mC, vCostFun ] = ClusterKMeans( mX, mC, hDistFun, numIterations, stopTol, dispFig, saveFig, dispTime )
% ----------------------------------------------------------------------------------------------- %
% [ vClusterIdx, mC, vCostFun ] = ClusterKMeans( mX, mC, hDistFun, numIterations, stopTol, dispFig, saveFig, dispTime )
%   Vanilla implementation of the K-Means algorithm with support for any
%   distance function to generate the Distance Matrix.
% Input:
%   - mX            -   Data Matrix.
%                       Each data sample is a row of the matrix.
%                       Structure: Matrix (numVars x varDim).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - mD            -   Distance Matrix.
%                       A symmetric matrix where 'mD(ii, jj) = dist(mX(ii,
%                       :), mX(jj, :))';.
%                       Structure: Matrix (numVars x numVars).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     01/01/2021  Royi Avital	RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    mX (:, :) {mustBeNumeric, mustBeReal}
    mC (:, :) {mustBeNumeric, mustBeReal}
    hDistFun (1, 1) {mustBeFunctionHandler}
    numIterations (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger}
    stopTol (1, 1) {mustBeNumeric, mustBeReal, mustBePositive}
    dispFig (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeInRange(dispFig, 0, 1)} = 0
    saveFig (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeInRange(saveFig, 0, 1)} = 0
    dispTime (1, 1) {mustBeNumeric, mustBeReal, mustBeInRange(dispTime, 0, 10)} = 1
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

if(dispFig)
    figureIdx           = 0;
    if(saveFig)
        figureCounterSpec   = '%04d';
    end
end

numClsuters = size(mC, 2);
% numDim      = size(mC, 1);
% numSamples  = size(mX, 1);

vCostFun    = -ones(numIterations, 1);

ii = 1;

mD                      = hDistFun(mX, mC);
[vMinDist, vClusterIdx] = min(mD, [], 2);
vCostFun(ii)            = sum(vMinDist);

if(dispFig)
    figureIdx = figureIdx + 1;
    [hF, hA] = DisplayClusterData(mX, mC, vClusterIdx, ii, vCostFun(ii));
    if(saveFig)
        print(hF, ['FigureKMeans', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
    end
    if(dispTime > 0)
        pause(dispTime);
    end
    delete(hF);
end

for ii = 2:numIterations
    
    for jj = 1:numClsuters
        mC(:, jj) = mean(mX(:, vClusterIdx == jj), 2);
    end
    
    mD                      = hDistFun(mX, mC);
    [vMinDist, vClusterIdx] = min(mD, [], 2);
    vCostFun(ii)            = sum(vMinDist);
    
    if(dispFig)
        figureIdx = figureIdx + 1;
        [hF, hA] = DisplayClusterData(mX, mC, vClusterIdx, ii, vCostFun(ii));
        if(saveFig)
            print(hF, ['FigureKMeans', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
        end
        if(dispTime > 0)
            pause(dispTime);
        end
        delete(hF);
    end
    
    if(abs(vCostFun(ii) - vCostFun(ii - 1)) < stopTol)
        break;
    end
    
end


end


function [ ] = mustBeFunctionHandler( hF )
% https://www.mathworks.com/matlabcentral/answers/107552
if ~isa(hF, 'function_handle')
    eid = 'mustBeFunctionHandler:notFunctionHandler';
    msg = 'The 2nd input must be a Function Handler';
    throwAsCaller(MException(eid, msg));
end


end

