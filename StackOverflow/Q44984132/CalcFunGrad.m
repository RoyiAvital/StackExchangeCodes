function [ vG ] = CalcFunGrad( vX, hObjFun, difMode, epsVal )
% ----------------------------------------------------------------------------------------------- %
% [ vG ] = CalcFunGrad( vX, hObjFun, difMode, epsVal )
%   Calculating the Gradient Vector of a function using Finite Differences
%   method.
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
%   - difMode       -   Difference Mode.
%                       Sets the mode of operation of the Finite
%                       Differences Method - Forward, Backward or Central.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3}.
% Output:
%   - vG            -   Gradient Vector.
%                       The numerical approximation of the gradient of the
%                       Objective Function at the input point 'vX'.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://en.wikipedia.org/wiki/Finite_difference_coefficient.
%   2.  https://stackoverflow.com/a/43099198/195787.
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     08/07/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vX, 1);

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;

objFunRef   = hObjFun(vX);
vG          = zeros([numElements, 1]);
vPertVal    = zeros([numElements, 1]);

switch(difMode)
    case(DIFF_MODE_FORWARD)
        hCalcGradFun = @(vPertVal) (hObjFun(vX + vPertVal) - objFunRef) / epsVal;
    case(DIFF_MODE_BACKWARD)
        hCalcGradFun = @(vPertVal) (objFunRef - hObjFun(vX - vPertVal)) / epsVal;
    case(DIFF_MODE_CENTRAL)
        hCalcGradFun = @(vPertVal) (hObjFun(vX + vPertVal) - hObjFun(vX - vPertVal)) / (2 * epsVal);
end

for ii = 1:numElements
    vPertVal(ii)    = epsVal;
    vG(ii)          = hCalcGradFun(vPertVal);
    vPertVal(ii)    = 0;
end


end

