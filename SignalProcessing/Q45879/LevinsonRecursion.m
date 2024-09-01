function [ vX ] = LevinsonRecursion( mT, vY )
% ----------------------------------------------------------------------------------------------------- %
%[ vX ] = LevinsonRecursion( mT, vY )
% Solves Linear System mT * vX = vY where the Matrix 'mT' is a Toeplitz
% Matrix using Levinson Recursion improved speed (Complexity of N^2 instead
% of N^3). Input:
%   - mT    -   Input Matrix.
%               Matrix with a Toeplitz Structure.
%               Structure: Matrix (N x N).
%               Type: 'Single' / 'Double'.
%               Range: (-inf, inf).
%   - vY    -   Input Vector.
%               Structure: Column Vector (N x 1).
%               Type: 'Single' / 'Double'.
%               Range: (-inf, inf).
% Output:
%   - vX    -   Output Vector.
%               The solution of the system mT * vX = vY.
%               Structure: Column Vector (N x 1).
%               Type: 'Single' / 'Double'.
%               Range: (-inf, inf).
% References
%   1.  Levinson Recursion (Wikipedia) - https://en.wikipedia.org/wiki/Levinson_recursion
% Remarks:
%   1.  Joint work with Yair Shemer.
%   2.  This implementation supports only rectangular matrices.
% TODO:
%   1.  A
% Release Notes:
%   -   1.0.000     15/02/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------------- %

numRows = size(mT, 1); %<! Square Matrix

vF = zeros([numRows, 1]); %<! Forward Vector
vB = zeros([numRows, 1]); %<! Backward Vector
vX = zeros([numRows, 1]); 

% Initialized vF, vB and vX;
vF(1) = 1 / mT(1, 1); 
vB(1) = 1 / mT(1, 1); 
vX(1) = vY(1) / mT(1, 1);

for iRow = 2:numRows
    % Calculate the epsilons (error expressions)
    epsF = mT(iRow, 1:(iRow - 1)) * vF(1:(iRow - 1));
    epsB = mT(1, 2:iRow) * vB(1:(iRow - 1));
    epsX = mT(iRow, 1:(iRow - 1)) * vX(1:(iRow - 1));
    
    % Given vF, vB and vX with size n-1 find those vectors with size n
    vFPrev      = vF;
    scalingFctr = 1 / (1 - (epsF * epsB));
    
    vF(1:iRow) = scalingFctr * vF(1:iRow) - (epsF * scalingFctr) * [0; vB(1:(iRow - 1))];
    vB(1:iRow) = scalingFctr * [0; vB(1:(iRow - 1))] - (epsB * scalingFctr) * vFPrev(1:iRow);
    vX(1:iRow) = vX(1:iRow) + (vY(iRow) - epsX) * vB(1:iRow);
end


end

