function [ vX ] = SolveRidgeReg( mA, vB, paramLambda, sOptParams )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = SolveLsIteratively( mA, vB, paramLambda, sOptParams )
% Solve the problem:
%   \arg \min_{x} || A x - b ||_{2}^{2} + lambda * || x ||_{2}^{2}
% The solution is done iteratively in order to support large sparse
% matrices.
% Input:
%   - mA                -   Input Matrix.
%                           Structure: Matrix (numRows x numCols).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vB                -   Input Vector.
%                           The model known data.
%                           Structure: Vector (numRows X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf). 
%   - stopTol           -   Stop Tolerance.
%                           Sets the accuracy of the solution. Lower values
%                           means more iterations are required to get the
%                           precision required.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - maxNumIter        -   Maximum Number of Iterations.
%                           The maximum number of iterations allowed to the
%                           solver to reach within the required tolerance.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3}.
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (numCols X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  The solution is by solveing the following least squares problem:
%       \arg \min_{x} || [A; sqrt(lambda) * I] * x - [b; 0] ||_{2}^{2}
%   2.  B
% TODO:
%   1.
%   Release Notes:
%   -   1.0.000     18/07/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    mA (:, :) {mustBeNumeric, mustBeReal}
    vB (:, 1) {mustBeNumeric, mustBeReal}
    paramLambda {mustBeNumeric, mustBeReal, mustBeNonnegative} = 0
    sOptParams.stopTol (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1e-7
    sOptParams.maxNumIter (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 100
end

stopTol     = sOptParams.stopTol;
maxNumIter  = sOptParams.maxNumIter;

[numRows, numCols] = size(mA);

if(paramLambda == 0)
    vX = lsqr(mA, vB, stopTol, maxNumIter);
else
    vX = lsqr([mA; sqrt(paramLambda) * speye(numCols)], [vB; zeros(numCols, 1)], stopTol, maxNumIter);
end


end

