function [ vX ] = SolveLsL0Bomp( mA, vB, numBlocks, paramK, tolVal )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveLsL0Omp( mA, vB, paramK, tolVal )
% Minimizes Least Squares of Linear System with L0 Constraint Using
% Block Orthogonal Matching Pursuit (OMP) Method.
% \arg \min_{x} {\left\| A x - b \right\|}_{2}^{2} subject to {\left\| x
% \right\|}_{2, 0} \leq K
% Input:
%   - mA                -   Input Matrix.
%                           The model matrix (Fat Matrix). Assumed to be
%                           normalized. Namely norm(mA(:, ii)) = 1 for any
%                           ii.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'. 
%                           Range: (-inf, inf).
%   - vB                -   input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - numBlocks         -   Number of Blocks.
%                           The number of blocks in the problem structure.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}.
%   - paramK            -   Parameter K.
%                           The L0 constraint parameter. Basically the
%                           maximal number of active blocks in the
%                           solution.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}.
%   - tolVal            -   Tolerance Value.
%                           Tolerance value for equality of the Linear
%                           System.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range [0, inf).
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  An Optimal Condition for the Block Orthogonal Matching Pursuit
%       Algorithm - https://ieeexplore.ieee.org/document/8404118.
%   2.  Block Sparsity: Coherence and Efficient Recovery - https://ieeexplore.ieee.org/document/4960226.
% Remarks:
%   1.  The algorithm assumes 'mA' is normalized (Each column).
%   2.  The number of columns in matrix 'mA' must be an integer
%       multiplication of the number of blocks.
%   3.  For 'numBlocks = numColumns' (Equivalent of 'numElmBlock = 1') the
%       algorithm becomes the classic OMP.
% Known Issues:
%   1.  A
% TODO:
%   1.  Pre Process 'mA' by normalizing its columns.
% Release Notes:
%   -   1.0.000     19/08/2019
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

numRows = size(mA, 1);
numCols = size(mA, 2);

numElmBlock = numCols / numBlocks;
if(round(numElmBlock) ~= numElmBlock)
    error('Number of Blocks Doesn''t Match Size of Arrays');
end

vActiveIdx      = false([numCols, 1]);
vR              = vB;
vX              = zeros([numCols, 1]);
activeBlckIdx   = [];

for ii = 1:paramK
    
    maxCorr         = 0;
    
    for jj = 1:numBlocks
        vBlockIdx = (((jj - 1) * numElmBlock) + 1):(jj * numElmBlock);
        
        currCorr = abs(mA(:, vBlockIdx).' * vR);
        if(currCorr > maxCorr)
            activeBlckIdx = jj;
            maxCorr = currCorr;
        end
    end
    
    vBlockIdx = (((activeBlckIdx - 1) * numElmBlock) + 1):(activeBlckIdx * numElmBlock);
    vActiveIdx(vBlockIdx) = true();
    
    vX(vActiveIdx) = mA(:, vActiveIdx) \ vB;
    vR = vB - (mA(:, vActiveIdx) * vX(vActiveIdx));
    
    resNorm = norm(vR);
    
    if(resNorm < tolVal)
        break;
    end
    
end


end

