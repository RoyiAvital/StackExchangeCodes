function [ vX ] = SolveBasisPursuitLp001( mA, vB )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveBasisPursuitLp002( mA, vB )
% Solve Basis Pursuit problem using Linear Programming.
% Input:
%   - mA                -   Input Matirx.
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
%   -   1.0.000     29/03/2018
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

numRows = size(mA, 1);
numCols = size(mA, 2);

%% vX = [vX; vT]

mAeq = [mA, zeros(numRows, numCols)];
vBeq = vB;

vF = [zeros([numCols, 1]); ones([numCols, 1])];
mA = [eye(numCols), -eye(numCols); -eye(numCols), -eye(numCols)];
vB = zeros(2 * numCols, 1);

sSolverOptions = optimoptions('linprog', 'Display', 'off');
vX = linprog(vF, mA, vB, mAeq, vBeq, [], [], sSolverOptions);
vX = vX(1:numCols);


end