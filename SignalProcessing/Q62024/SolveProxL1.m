function [ vX ] = SolveProxL1( vY, paramLambda )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = SolveProxL1( vY, paramLambda )
% Solves the Prox of the L1 Norm using Alternating direct method, the Soft
% Threshold operator. Basically solves the problem given by:
% $$ \arg \min_{ x \in \mathbb{R}^{n} } \frac{1}{2} {\left\| x - y \right|}_{2}^{2} + \lambda {\left\| x \right\|}_{1} $$
% Input:
%   - vY                -   Input Vector.
%                           The vector to apply the PRox operator upon.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The L1 Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - vX                -   Output Vector.
%                           Structure: Vector (n X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% Known Issues:
%   1.  C
% TODO:
%   1.  Pre calculate decomposition of the Linear System.
% Release Notes:
%   -   1.0.000     27/11/2019  Royi Avital
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %



% Soft Thresholding
vX = max(vY - paramLambda, 0) + min(vY + paramLambda, 0);


end

