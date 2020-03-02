function [ vX ] = SolveL1NormSetMinimization( mY )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mX ] = SolveLsBoxConstraints( mA, vB, vC, vD, vX, numIterations, stopTol )
%   Solves \arg \min_{x} 0.5 || A x - b ||, s.t. c <= x <= d using
%   Gradient Descent Method.
% Input:
%   - mA            -   Model Matrix.
%                       Input model matrix.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem..
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     02/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vecDim      = size(mY, 1);
numSamples  = size(mY, 2);

% vX = [vX; vT]
numElements = vecDim * numSamples;

% Summing over all {t}_{ij}
vF = [zeros(vecDim, 1); ones(numElements, 1)];

% Building the matrix where we run on {x}_{i}, {t}_{i, j}, {y}_{i, j} in
% column wise manner. Where i is the element index and j is the sample
% index.
mA = [repmat(speye(vecDim), numSamples, 1), -speye(numSamples * vecDim); repmat(-speye(vecDim), numSamples, 1), -speye(numSamples * vecDim)];
vB = [mY(:); -mY(:)];

sSolverOptions = optimoptions('linprog', 'Display', 'off');
vX = linprog(vF, mA, vB, [], [], [], [], sSolverOptions);
vX = vX(1:vecDim);


end

