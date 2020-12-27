function [ vX, mX ] = ProjectedGd( vX, hG, hP, numIterations, stepSize )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mX ] = ProjectedGd( vX, hG, hP, numIterations )
% Solves an objective function by the Projected Gradient Descent /
% Projected Gradient Method (PGM).
% Input:
%   - vX                -   Input Vector.
%                           The starting point for the gradient iterations.
%                           Structure: Vector (numElements x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - hG                -   Gradient Function Handler.
%                           A function handler which accepts a vector of
%                           size (numElements x 1) and returns the gradient
%                           in the form of a vector (numElements x 1).
%                           Structure: Function Handler.
%                           Type: NA.
%                           Range: NA.
%   - hP                -   Gradient Function Handler.
%                           A function handler which accepts a vector of
%                           size (numElements x 1) and returns the gradient
%                           in the form of a vector (numElements x 1).
%                           Structure: Function Handler.
%                           Type: NA.
%                           Range: NA.
%   - numIterations     -   Number of Iterations.
%                           Sets the number of iterations of the gradient
%                           descent.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, 3, ...}.
%   - stepSize          -   Step Size.
%                           Sets the step size of each iteration.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - vX                -   Output Vector.
%                           The end point for the gradient iterations.
%                           Structure: Vector (numElements x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mX                -   The Path Matrix.
%                           The points of the gradient iterations.
%                           Structure: Matrix (numElements x numIterations).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
%   Release Notes:
%   -   1.0.000     26/12/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

mX = zeros(size(vX, 1), numIterations);

mX(:, 1) = vX;

for ii = 2:numIterations
    vX(:) = vX - stepSize * hG(vX); %<! Gradient Step
    vX(:) = hP(vX); %<! Projection Step
    
    mX(:, ii) = vX;
end


end

