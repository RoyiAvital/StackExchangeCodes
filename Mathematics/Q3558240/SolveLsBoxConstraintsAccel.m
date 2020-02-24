function [ vX, mX ] = SolveLsBoxConstraintsAccel( mA, vB, vC, vD, vX, numIterations, stopTol )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mX ] = SolveLsBoxConstraintsAccel( mA, vB, vC, vD, vX, numIterations, stopTol )
%   Solves \arg \min_{x} 0.5 || A x - b ||, s.t. c <= x <= d using
%   accelerated Gradient Descent Method.
% Input:
%   - mA            -   Model Matrix.
%                       Input model matrix.
%                       Structure: Matrix (m x n).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vB            -   Measurements Vector.
%                       Given data vector.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vC            -   Box Constraints Vector (Minimum Value).
%                       Sets the minimum value for the solution.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - vD            -   Box Constraints Vector (Maximum Value).
%                       Sets the maximum value for the solution.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations for the algorithm to
%                       run.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range {1, 2, ...}.
%   - stopTol       -   Stopping Condition Tolerance.
%                       Sets the stopping threshold for the L Inf (Maximum
%                       Absolute Value) of the change between 2 iterations
%                       of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range [0, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem..
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - mX            -   Solution Path Matrix.
%                       Matrix which each of its columns embeds the
%                       solution of the i-th step.
%                       Structure: Matrix (m x numIterations).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  If one sets 'stopTol = 0' then the algorithm will run the given
%       number of iterations.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     24/02/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

FISTA_STEP__METHOD_SMOOTH       = 1; %<! Monotonic, Slower
FISTA_STEP__METHOD_AGGRESSIVE   = 2; %<! Non Monotonic, Faster

% vX = mA \ vB;
% 
% if(all(vX >= vC) && all(vX <= vD))
%     return;
% end

fistaStepMode = FISTA_STEP__METHOD_SMOOTH;

nDim = size(mA, 2);

mAA = mA.' * mA;
vAb = mA.' * vB;

% Lipschitz Constant
stepSize = 1 / (2 * (norm(mA, 2) ^ 2));
% stepSize = 1 / sum(mA(:) .^ 2); %<! Faster to calculate, conservative (Hence slower)

vY = vX;
tK = 1;
mX = zeros(nDim, numIterations);
mX(:, 1) = vX;

for ii = 2:numIterations
    
    vXPrev  = vX;
    
    vG      = (mAA * vY) - vAb;
    vY      = vY - (stepSize * vG);
    vX      = min(max(vC, vY), vD);
    
    switch(fistaStepMode)
        case(FISTA_STEP__METHOD_SMOOTH)
            tKPrev          = tK;
            tK              = (1 + sqrt(1 + (4 * tKPrev))) / 2;
            fistaStepSize   = (tKPrev - 1) / tK;
        case(FISTA_STEP__METHOD_AGGRESSIVE)
            fistaStepSize = (ii - 1) / (ii + 2);
    end
    
    vY = vX + (fistaStepSize * (vX - vXPrev));
    
    mX(:, ii) = vX;
    
    if(max(abs(vX - mX(:, ii - 1))) <= stopTol)
        break;
    end
end


end

