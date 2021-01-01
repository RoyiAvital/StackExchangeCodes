function [ hF, hA ] = DisplayClusterData( mX, mC, vClusterIdx, itrIdx, costFunVal )
% ----------------------------------------------------------------------------------------------- %
% [ hF, hA ] = DisplayClusterData( mX, mC, vClusterIdx, itrIdx, costFunVal )
%   Displayes clustering data.
% Input:
%   - mA            -   Data Matrix.
%                       Each data sample is a row of the matrix.
%                       Structure: Matrix (numVarsA x varDim).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mB            -   Data Matrix.
%                       Each data sample is a row of the matrix.
%                       Structure: Matrix (numVarsB x varDim).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - mD            -   Distance Matrix.
%                       A symmetric matrix where 'mD(ii, jj) = dist(mA(ii,
%                       :), mB(jj, :));'.
%                       Structure: Matrix (numVarsA x numVarsB).
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
    mX (2, :) {mustBeNumeric, mustBeReal}
    mC (2, :) {mustBeNumeric, mustBeReal}
    vClusterIdx (:, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger}
    itrIdx (1, 1) {mustBeNumeric, mustBeReal, mustBeNonnegative, mustBeInteger} = 0 %<! Non Negative to allow 0 to exclude
    costFunVal (1, 1) {mustBeNumeric, mustBeReal, mustBeNonnegative} = 0 %<! Non Negative to allow 0 to exclude
%     lineWidthSample (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 3
%     markerEdgeSample (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 0.4
%     lineWidthCentroid (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 4
%     sizeDataCentroid (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 256
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

lineWidthSample     = 3;
markerEdgeSample    = 0.15;
lineWidthCentroid   = 4;
sizeDataCentroid    = 256;
fontSizeTitle       = 14;

numClusters = size(mC, 2);
numDim      = size(mC, 1);
numSamples  = size(mX, 1);

mColorOrder = lines(numClusters);

% hFigure = figure('Position', [100, 100, 800, 800]);
hF = figure();
hA   = axes('DataAspectRatio', [1, 1, 1], 'NextPlot', 'add');

for ii = 1:numClusters
    
    scatterIdx = (2 * ii) - 1;
    
    hScatterObj(scatterIdx) = scatter(mX(1, vClusterIdx == ii), mX(2, vClusterIdx == ii));
    set(hScatterObj(scatterIdx), 'MarkerEdgeColor', mColorOrder(ii, :), 'LineWidth', lineWidthSample, 'MarkerEdgeAlpha', markerEdgeSample);
    
    scatterIdx = 2 * ii;
    
    hScatterObj(scatterIdx) = scatter(mC(1, ii), mC(2, ii));
    set(hScatterObj(scatterIdx), 'MarkerEdgeColor', mColorOrder(ii, :), 'Marker', '+', 'SizeData', sizeDataCentroid, 'LineWidth', lineWidthCentroid);
    
end

set(hA, 'LooseInset', [0.07, 0.07, 0.07, 0.07]);

if(itrIdx > 0)
    sItrIdx = ['Iteration #', num2str(itrIdx, '%05d')];
end
if(costFunVal > 0)
    sCostFunVal = ['Objective Function Value - ', num2str(costFunVal)];
end

if((itrIdx > 0) && (costFunVal > 0))
    sItrData = [sItrIdx, ', ', sCostFunVal];
elseif(itrIdx > 0)
    sItrData = sItrIdx;
elseif(costFunVal > 0)
    sItrData = sCostFunVal;
else
    sItrData = [];
end

sFigTitle = {['K - Means Clusters']};
if(~isempty(sItrData))
    sFigTitle{2} = sItrData;
end

set(get(hA, 'Title'), 'String', sFigTitle, 'Fontsize', fontSizeTitle);


end

