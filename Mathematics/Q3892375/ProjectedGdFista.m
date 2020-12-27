function [ vX, vV, mX ] = ProjectedGdFista( vX, vV, hG, hP, numIterations, stepSize )
% ----------------------------------------------------------------------------------------------- %
% [ vX, vV, mX ] = ProjectedGdFista( vX, vV, hG, hP, numIterations, stepSize )
% Solves an objective function by the Projected Gradient Descent /
% Projected Gradient Method (PGM) with FISTA Acceleration.
% Input:
%   - vX                -   Input Vector.
%                           The starting point for the gradient iterations.
%                           Structure: Vector (numElements x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vV                -   Acceleration Vector.
%                           The buffer to hold the accelerated direction
%                           vector.
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
%                           Sets the step size of the Gradient step.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - vX                -   Output Vector.
%                           The end point for the gradient iterations.
%                           Structure: Vector (numElements x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vV                -   Acceleration Vector.
%                           The buffer to hold the accelerated direction
%                           vector of last iteration. May used by a wrapper
%                           for warm start.
%                           Structure: Vector (numElements x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - mX                -   The Path Matrix.
%                           The points of the gradient iterations.
%                           Structure: Matrix (numElements x numIterations).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  Geoff Gordon & Ryan Tibshirani - Accelerated First Order Methods (https://www.cs.cmu.edu/~ggordon/10725-F12/slides/09-acceleration.pdf).
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     26/12/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

mX = zeros(size(vX, 1), numIterations);

vG          = zeros(size(vX, 1), 1);
mX(:, 1)    = vX;
mX(:, 2)    = vX - stepSize * hG(vX);

for ii = 3:numIterations
    vV(:) = mX(:, ii - 1) + (((ii - 2) / (ii + 1)) * (mX(:, ii - 1) - mX(:, ii - 2)));
    vG(:) = hG(vV); %<! The gradient
    vX(:) = hP(vV - (stepSize * vG)); %<! Projection step
    
    mX(:, ii) = vX;
end


end

