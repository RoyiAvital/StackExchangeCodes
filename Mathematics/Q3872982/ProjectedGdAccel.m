function [ vX, vV, mX ] = ProjectedGdAccel( vX, vV, hG, hP, numIterations, stepSizeGd, stepSizeAccel )
% ----------------------------------------------------------------------------------------------- %
% [ vX, vV, mX ] = ProjectedGdAccel( vX, vV, hG, hP, numIterations, stepSizeGd, stepSizeAccel )
% Solves an objective function by the Projected Gradient Descent /
% Projected Gradient Method (PGM) with Nesterov Acceleration Gradient
% (NAG).
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
%   - stepSizeGd        -   Step Size.
%                           Sets the step size of the Gradient step.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - stepSizeAccel     -   Acceleration Step Size.
%                           The step size of the Nesterov Acceleration
%                           step.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1).
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
%   1.  A
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

for ii = 2:numIterations
    vG(:) = hG(vX - (stepSizeAccel * vV)); %<! The look ahead gradient
    vV(:) = (stepSizeAccel * vV) + (stepSizeGd * vG); %<! Momentum step
    vX(:) = hP(vX - vV); %<! Projection step
    
    mX(:, ii) = vX;
end


end

