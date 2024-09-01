function [ vX ] = SolveLsL0Omp( mA, vB, paramK, tolVal )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveLsL0Omp( mA, vB, paramK, tolVal )
% Minimizes Least Squares of Linear System with L0 Constraint Using
% Orthogonal Matching Pursuit (OMP) Method.
% \arg \min_{x} {\left\| A x - b \right\|}_{2}^{2} subject to {\left\| x
% \right\|}_{0} \leq K
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
%   - paramK            -   Parameter K.
%                           The L0 constraint parameter.
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
%   1.  Wikipedia MP - https://en.wikipedia.org/wiki/Matching_pursuit.
%   2.  Michael Elad - Sparse and Redundant Representations (Pages 36-40)
% Remarks:
%   1.  The algorithm assumes 'mA' is normalized (Each column).
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

vActiveIdx  = false([numCols, 1]);
vR          = vB;
vX          = zeros([numCols, 1]);

for ii = 1:paramK
    
    % Maximum Correlation minimizes the L2 of the Error given atoms are
    % normalized
    [~, activeIdx] = max(abs(mA.' * vR));
    
    vActiveIdx(activeIdx) = true();
    
    vX(vActiveIdx) = mA(:, vActiveIdx) \ vB;
    vR = vB - (mA(:, vActiveIdx) * vX(vActiveIdx));
    
    resNorm = norm(vR);
    
    if(resNorm < tolVal)
        break;
    end
    
end


end

