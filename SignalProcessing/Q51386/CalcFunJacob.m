function [ mJ ] = CalcFunJacob( vX, hObjFun, difMode, epsVal )
% ----------------------------------------------------------------------------------------------- %
% [ vG ] = CalcFunJacob( vX, hObjFun, difMode, epsVal )
%   Calculating the Jacobian Matrix of a function using Finite Differences
%   method.
% Input:
%   - vX            -   Input Vector.
%                       The point the Jacobian is caclaulated at.
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
%                       Differences Method - Forward, Backward, Central or
%                       Complex.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, 4}.
% Output:
%   - mJ            -   Jacobian Matrix.
%                       The numerical approximation of the Jacobian of the
%                       Objective Function at the input point 'vX'.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://en.wikipedia.org/wiki/Finite_difference_coefficient.
% Remarks:
% Remarks:
%   1.  If the Complex Mode is selected the function must return complex
%       values in order to work. For instance, if the input function is
%       'norm(vX)' use 'sum(vX .^ 2)' and if the input function is
%       'sum(abs(vX))' use 'sum(sqrt(vX .^ 2))'.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     22/08/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;


vObjFunRef  = hObjFun(vX);

numRows = size(vObjFunRef, 1);
numCols = size(vX, 1);

mJ          = zeros(numRows, numCols);
vPertVal    = zeros([numCols, 1]);

switch(difMode)
    case(DIFF_MODE_FORWARD)
        hCalcGradFun = @(vPertVal) (hObjFun(vX + vPertVal) - vObjFunRef) / epsVal;
    case(DIFF_MODE_BACKWARD)
        hCalcGradFun = @(vPertVal) (vObjFunRef - hObjFun(vX - vPertVal)) / epsVal;
    case(DIFF_MODE_CENTRAL)
        hCalcGradFun = @(vPertVal) (hObjFun(vX + vPertVal) - hObjFun(vX - vPertVal)) / (2 * epsVal);
    case(DIFF_MODE_COMPLEX)
        hCalcGradFun = @(vPertVal) imag(hObjFun(vX + (1i * vPertVal))) / epsVal;
end

for ii = 1:numCols
    vPertVal(ii)    = epsVal;
    mJ(:, ii)       = hCalcGradFun(vPertVal);
    vPertVal(ii)    = 0;
end


end

