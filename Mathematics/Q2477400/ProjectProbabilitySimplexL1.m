function [ vX ] = ProjectProbabilitySimplexL1( vY )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectProbabilitySimplexL1( vY )
%   Solves \arg \min_{x} || x - y ||_{1} s.t. x \in Probability Simplex
%   using reformulation of the problem as Linear Programming Problem.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem.
%                       Structure: Vector (n x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  See https://math.stackexchange.com/questions/2477400.
% Remarks:
%   1.  The solution reformulate the L1 Norm with auxiliary variables 'vT'.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     13/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);

% The model variable is [vT; vX];
numVarsT = numElements;
numVarsX = numElements;
numVars = numVarsT + numVarsX;

% Model
vF = zeros(numVars, 1);
vF(1:numVarsT) = 1; %<! Sum of 'vT'

% Inequlaity Constraints
mA = [-speye(numVarsT), speye(numVarsX); -speye(numVarsT), -speye(numVarsX)];
vB = [vY; -vY];

% Equality Constraints
vA = zeros(1, numVars); %<! Row in a Matrix
vA((numVarsT + 1):numVars) = 1; %<! Summing vX variables
valB = 1;

% Bounds
vL = zeros(numVars, 1);
vU = inf(numVars, 1);

% Solver
sSolverOptions = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'off'); 

vTX = linprog(vF, mA, vB, vA, valB, vL, vU, sSolverOptions);
vX  = vTX((numVarsT + 1):numVars); %<! Extracting 'vX' from '[vT; vX];'.


end

