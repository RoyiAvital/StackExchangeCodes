function [ vX ] = SolveBasisPursuitLp002( mA, vB )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveBasisPursuitLp001( mA, vB )
% Solve Basis Pursuit problem using Linear Programming.
% Input:
%   - mA                -   Input Matrix.
%                           The model matrix.
%                           Structure: Matrix (m X n).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   input Vector.
%                           The model known data.
%                           Structure: Vector (m X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  Wikipedia Basis Pursuit - https://en.wikipedia.org/wiki/Basis_pursuit.
% Remarks:
%   1.  A
% Known Issues:
%   1.  A
% TODO:
%   1.  Implement in the form of course.
% Release Notes:
%   -   1.1.000     31/03/2018
%       *   Changed form to match class form.
%   -   1.0.000     29/03/2018
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

numRows = size(mA, 1);
numCols = size(mA, 2);

% vU = max(vX, 0);
% vV = max(-vX, 0);
% vX = vU - vX;
% vUV = [vU; vV];

vF = ones([2 * numCols, 1]);

mAeq = [mA, -mA];
vBeq = vB;

vLowerBound = zeros([2 * numCols, 1]);
vUpperBound = inf([2 * numCols, 1]);

sSolverOptions = optimoptions('linprog', 'Display', 'off');

vUV = linprog(vF, [], [], mAeq, vBeq, vLowerBound, vUpperBound, sSolverOptions);

vX = vUV(1:numCols) - vUV(numCols + 1:end);


end