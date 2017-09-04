function [ vX ] = ProjectOntoLinearInequality( vY, mC, vD, stopThr )
% ----------------------------------------------------------------------------------------------- %
%[ vX ] = ProjectOntoLinearInequality( vY, mC, vD, stopThr )
% Applies the projection onto the Convex Polytop / Convex Polyhedron: mC *
% vX <= vD using Alternating Minimzation. The problem is formulated as:
% \arg \min_{x} \frac{1}{2} \left\| x - y \right\|_{2}^{w} subject to C x
% <= d.
% Input:
%   - vD                -   Input Vector.
%                           The vector to be projected onto the polyhedron.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%   - mC                -   Polyhedron Matrix.
%                           The given matrix of the polyhedron.
%                           Structure: Vector (Column Vector).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vD                -   Polyhedron Vector.
%                           The given vector of the polyhedron.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}.
%   - stopThr           -   Stopping Threshold.
%                           Threshold for stopping the Alternating
%                           Projection.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - vX                -   Projected Vector.
%                           The optimal solution of the problem 0.5 || x
%                           - y || ^ 2 subject to {c}^{T} x <= d.
%                           Structure: Vector (Column Vector).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  Projections onto Convex Sets (Wikipedia) - https://en.wikipedia.org/wiki/Projections_onto_convex_sets.
%   2.  Convex Polytope (Wikipedia) - https://en.wikipedia.org/wiki/Convex_polytope.
%   3.  Derivation (StackExchange Mathematics) - https://math.stackexchange.com/a/2416843/33.
% Remarks:
%   1.  T
% TODO:
%   1.  U
% Release Notes:
%   -   1.0.000     04/09/2017  Royi Avital
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

numConst    = size(mC, 1);
% Pre calculation
vCNorm      = sum(mC .^ 2, 2);

vX      = vY;
vRes    = (mC * vX) - vD;
maxRes  = max(vRes);

while(maxRes > stopThr)
    for ii = 1:numConst
        paramLambda = max(((mC(ii, :) * vX) - vD(ii)) / vCNorm(ii), 0);
        vX = vX - (paramLambda * mC(ii, :).');
    end
    
    vRes = (mC * vX) - vD;
    maxRes = max(vRes);
    
end


end

