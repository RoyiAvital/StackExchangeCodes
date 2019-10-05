function [ mW ] = CalcSpatialWeights( kernelRadius, spatialStd )
% ----------------------------------------------------------------------------------------------- %
% [ mW ] = CalcSpatialWeights( kernelRadius, spatialStd )
%   Calculates the Spatial Weights of Bilateral Filter.
% Input:
%   - vX            -   Input Vector.
%                       The point the gradient is caclaulated at.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - hObjFun       -   Objective Function.
%                       Function handler which evaluates the Objective
%                       Function at a given point - hObjFun(vX).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - diffMode      -   Difference Mode.
%                       Sets the mode of operation of the Finite
%                       Differences Method - Forward, Backward, Central or
%                       Complex.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, 4}.
% Output:
%   - vG            -   Gradient Vector.
%                       The numerical approximation of the gradient of the
%                       Objective Function at the input point 'vX'.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C.
% Release Notes:
%   -   1.0.000     05/10/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

vX = [-kernelRadius:kernelRadius];

vW = exp(-(vX .* vX) / (2 * spatialStd * spatialStd));
mW = vW.' * vW;
mW = mW / sum(mW(:));


end

