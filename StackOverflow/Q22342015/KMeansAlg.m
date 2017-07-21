function [ vClusterId, mA ] = KMeansAlg( mX, mA, hDistFun, numIterations )
% ----------------------------------------------------------------------------------------------- %
% [ vClusterId, mA ] = KMeansAlg( mX, mA, hDistFun, numIterations )
%   Run the K-Means Algorithm with arbirary Distnace Function on the input
%   data.
% Input:
%   - mX            -   Input Data Samples.
%                       Each column is a data sampled of dimension D and
%                       the number of samples is N.
%                       Structure: Matrix (D x N).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mA            -   Data Centroids.
%                       Initialization of the Centroids for the K-Means
%                       algorithm where there are K Centroids.
%                       Structure: Matrix (D x K).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - hDistFun      -   Distance Function.
%                       Function handler which accetps 2 vectros of the
%                       same dimension and returns the distance (Scalar)
%                       between them.
%                       Structure: Function Handler.
%                       Type: NA.
%                       Range: NA.
%   - numIterations -   Number of Iterations.
%                       The number of iterations ot run the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
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
%   1.  https://en.wikipedia.org/wiki/K-means_clustering.
% Remarks:
%   1.  a
% TODO:
%   1.  Add stopping criteria for the case 'vClusterId' doesn't change.
% Release Notes:
%   -   1.0.000     21/07/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

dimOrder    = size(mX, 1);
numSamples  = size(mX, 2);
numClusters = size(mA, 2);

vClusterId = zeros([numClusters, 1]);


for ii = 1:numIterations
    
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
    
end


end

