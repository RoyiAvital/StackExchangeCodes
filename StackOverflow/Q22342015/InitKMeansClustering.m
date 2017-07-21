function [ vClusterId, mA ] = InitKMeansClustering( mX, numClusters, initMethod )
% ----------------------------------------------------------------------------------------------- %
% [ vClusterId, mA ] = InitKMeansClustering( mX, numClusters, initMethod )
%   Initialize teh Centroids and ID (Labels) for the K-Means algorithm.
% Input:
%   - mX            -   Input Data Samples.
%                       Each column is a data sampled of dimension D and
%                       the number of samples is N.
%                       Structure: Matrix (D x N).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numClusters   -   Number of Clusters.
%                       The number of clusters for the data.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ..., N}.
%   - initMethod    -   Initialization Method.
%                       Sets the method to initialize data. Currently
%                       either Random or by K-Means++ algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2}.
% Output:
%   - vClusterId    -   Cluster ID (Label).
%                       Vector with length of N where each data sample has
%                       ID (Label) of the cluster it belongs to.
%                       Structure: Vector (N x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ..., K}.
%   - mA            -   Data Centroids.
%                       The Centroids for the K-Means
%                       algorithm where there are K Centroids.
%                       Structure: Matrix (D x K).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://en.wikipedia.org/wiki/K-means%2B%2B.
% Remarks:
%   1.  a
% TODO:
%   1.  Complete the heuristic method (Similar to K-Means++ yet
%       deterministic).
% Release Notes:
%   -   1.0.000     21/07/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

INIT_METHOD_RANDOM              = 1;
INIT_METHOD_K_MEANS_PLUS_PLUS   = 2;
INIT_METHOD_HEURISTIC           = 3;

dimOrder    = size(mX, 1);
numSamples  = size(mX, 2);

mA = zeros([dimOrder, numClusters]);

switch(initMethod)
    case(INIT_METHOD_RANDOM)
        vClusterId = randi([1, numClusters], [numSamples, 1]);
        for jj = 1:numClusters
            mA(:, jj) = sum(mX(:, vClusterId == jj), 2) ./ sum(vClusterId == jj);
        end
    case(INIT_METHOD_K_MEANS_PLUS_PLUS)
        [vClusterId, mA] = InitKMeansPlusPlus(mX, mA);
    case(INIT_METHOD_HEURISTIC)
        vClusterId = randi([1, numClusters], [numSamples, 1]);
        for jj = 1:numClusters
            mA(:, jj) = sum(mX(:, vClusterId == jj), 2) ./ sum(vClusterId == jj);
        end
end


end


function [ vClusterId, mA ] = InitKMeansPlusPlus( mX, mA )

dimOrder    = size(mX, 1);
numSamples  = size(mX, 2);
numClusters = size(mA, 2);

mA(:, 1) = mX(:, randi([1, numSamples], [1, 1]));

vDistSqr = sum((mX - mA(:, 1)) .^ 2);

for ii = 2:numClusters
    
    selectedIdx = SelectWithoutReplacement(vDistSqr / sum(vDistSqr));
    mA(:, ii)   = mX(:, selectedIdx);
    
    % The Squared Distance to the closest point
    % Since 'vDistSqr' is the minimum from all pre chosen centers only
    % update regarding the new center.
    vDistSqr = min(vDistSqr, sum((mX - mA(:, ii)) .^ 2));
    
end

[~, vClusterId(:)] = min(sum(mA .^ 2, 1).' - (2 .* mA.' * mX), [], 1);


end

