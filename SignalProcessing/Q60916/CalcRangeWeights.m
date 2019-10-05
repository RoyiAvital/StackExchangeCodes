function [ mW ] = CalcRangeWeights( mI, vRefPixlCoord, kernelRadius, rangeStd )
% ----------------------------------------------------------------------------------------------- %
% [ mW ] = CalcRangeWeights( mI, vRefPixlCoord, kernelRadius, rangeStd )
%   Calculates the Range Weights of Bilateral Filter for specific patch.
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

kernlLength = (2 * kernelRadius) + 1;

mP = mI(vRefPixlCoord(1) - kernelRadius:vRefPixlCoord(1) + kernelRadius, vRefPixlCoord(2) - kernelRadius:vRefPixlCoord(2) + kernelRadius);
refPixelVal = mI(vRefPixlCoord(1), vRefPixlCoord(2));

mW = zeros(kernlLength, kernlLength);

for ii = 1:(kernlLength * kernlLength)
    mW(ii) = exp(-((mP(ii) - refPixelVal) ^ 2) / (2 * rangeStd * rangeStd));
end

mW = mW / sum(mW(:));


end

