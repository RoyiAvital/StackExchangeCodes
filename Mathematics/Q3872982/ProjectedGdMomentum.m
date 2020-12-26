function [ vX, vV, mX ] = ProjectedGdMomentum( vX, vV, hG, hP, numIterations, stepSizeGd, stepSizeMomentum )
% ----------------------------------------------------------------------------------------------- %
% [ vX, vV, mX ] = ProjectedGdMomentum( vX, vV, hG, hP, numIterations, stepSizeGd, stepSizeMomentum )
% Solves an objective function by the Projected Gradient Descent /
% Projected Gradient Method (PGM) with momentum.
% Input:
%   - vX                -   Input Vector.
%                           The starting point for the gradient iterations.
%                           Structure: Vector (numElements x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vV                -   Momentum Vector.
%                           The buffer to hold the momentum direction
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
%   - stepSizeMomentum  -   Momentum Step Size.
%                           The step size of the momentum step.
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
%                           The buffer to hold the momentum direction
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
%   1.  Why Momentum Really Works (https://distill.pub/2017/momentum/).
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

%{
V0 = 0
V1 = beta * v0 + alpha * (1 - beta) * vG1
V2 = beta * beta * v0 + beta * alpha * (1 - beta) * vG1 + alpha * (1 -
beta) * vG2 = alpha * beta * (1 - beta) * vG1 + alpha * (1 - beta) * vG2 =
alpha * (beta * (1 - beta) * vG1 + (1 - beta) * vG2)

Then:

vW = vW - Vt = vW - alpha * Vt`

Where Vt = beta * Vt-1 + (1 - beta) * vGt

%}
stepSize    = stepSizeGd * (1 - stepSizeMomentum);
vG          = zeros(size(vX, 1), 1);

mX(:, 1) = vX;

for ii = 2:numIterations
    vG(:) = hG(vX); %<! The gradient
    
    % My variant
%     vV(:) = (stepSizeMoment * vV) - (stepSizeGd * vG); %<! Momentum step
%     vX(:) = hP(vX + vV); %<! Projection step
    
    % https://towardsdatascience.com/a84097641a5d
    % Seems to have very insignificant improvement over SGD
%     vV(:) = (stepSizeMoment * vV) + ((1 - stepSizeMoment) * vG); %<! Momentum step
%     vX(:) = hP(vX - (stepSizeGd *  vV)); %<! Projection step
    
    % https://ruder.io/optimizing-gradient-descent/
    vV(:) = (stepSizeMomentum * vV) + (stepSizeGd * vG); %<! Momentum step
    vX(:) = hP(vX - vV); %<! Projection step
    
    mX(:, ii) = vX;
end


end

