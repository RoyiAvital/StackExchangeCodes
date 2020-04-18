function [ vX ] = SolveLsSameSign( mA, vB )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectProbabilitySimplexL1( vY )
%   Solves \arg \min_{x} || A x - b ||_{2}^{2} with the constraints that
%   all non zero values of x must have the same sign (Sign coordinated).
% Input:
%   - mA            -   Model Matrix.
%                       Structure: Vector (numRows x numCols).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vB            -   Input Vector.
%                       Structure: Vector (numRows x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem.
%                       Structure: Vector (numCols x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  See https://dsp.stackexchange.com/questions/52099.
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     19/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;


[vX1, resNorm1] = lsqnonneg(mA, vB);
[vX2, resNorm2] = lsqnonneg(-mA, vB);

if(resNorm1 < resNorm2)
    vX = vX1;
else
    vX = -vX2;
end


end

