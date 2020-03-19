function [ vX, mX ] = SolveLsUnitSimplexCgd( mA, vB, vX, numIterations, stopTol )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mX ] = SolveLsBoxConstraints( mA, vB, vC, vD, vX, numIterations, stopTol )
%   Solves \arg \min_{x} 0.5 || A x - b ||, s.t. 0 <= x, sum(x) = 1 using
%   Conditional Gradient Descent Method (Frank Wolfe Algorithm).
% Input:
%   - mA            -   Model Matrix.
%                       Input model matrix.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vB            -   Measurements Vector.
%                       Given data vector.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations for the algorithm to
%                       run.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range {1, 2, ...}.
%   - stopTol       -   Stopping Condition Tolerance.
%                       Sets the stopping threshold for the L Inf (Maximum
%                       Absolute Value) of the change between 2 iterations
%                       of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range [0, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem..
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mX            -   Solution Path Matrix.
%                       Matrix which each of its columns embeds the
%                       solution of the i-th step.
%                       Structure: Matrix (m x numIterations).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Frank Wolfe Algorithm (Wikipedia) - https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm.
%   2.  Notes on the Frank-Wolfe Algorithm, Part I - http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/.
%   3.  Notes on the Frank-Wolfe Algorithm, Part II - http://fa.bianp.net/blog/2018/fw2/.
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     19/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numRows = size(mA, 1);
numCols = size(mA, 2);

mAA = mA.' * mA;
vAb = mA.' * vB;

mX = zeros(numCols, numIterations);
mX(:, 1) = vX;

for ii = 2:numIterations
    vG = mAA * vX - vAb;
    
    [~, jj] = min(vG);
    stepSize = 2 / (2 + ii);
    
    % Frank Wolfe Step
    vX(:)   = vX - (stepSize * vX);
    vX(jj)  = vX(jj) + stepSize;
    
    mX(:, ii) = vX;
    
    if(max(abs(vX - mX(:, ii - 1))) <= stopTol)
        break;
    end
end


end

