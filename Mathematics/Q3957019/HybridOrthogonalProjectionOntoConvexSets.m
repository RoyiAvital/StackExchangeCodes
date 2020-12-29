function [ vX ] = HybridOrthogonalProjectionOntoConvexSets( cProjFun, vY, numIterations, stopThr, compMethod )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = HybridOrthogonalProjectionOntoConvexSets( cProjFun, vY, numIterations, stopThr, compMethod )
%   Solves \arg \min_{x} 0.5 || x - y ||, s.t. x \in \bigcap {C}_{i} using
%   Hybrid Orthogonal method.
% Input:
%   - cProjFun      -   Array of Projection Functions.
%                       Cell array of anonymouse functions which each is a
%                       projection into a sub space.
%                       Structure: Cell Array.
%                       Type: NA.
%                       Range: NA.
%   - vY            -   Input Vector.
%                       Input vector to be projected.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - numIterations -   Number of Iterations.
%                       Sets the number of iterations of the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ...}.
%   - stopThr       -   Stopping Threshold.
%                       Sets the threshold between consecutive iterations
%                       for stopping the algorithm.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - compMethod    -   Hybrid Method.
%                       Sets the hybrid method to calculate.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2}.
% Output:
%   - vX            -   Solution Vector.
%                       The solution to the optimization problem.
%                       Structure: Vector (m x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  Quadratic Optimization of Fixed Points of Non Expensive Mappings in Hilbert Space (https://doi.org/10.1080/01630569808816822).
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     19/03/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

COMP_METHOD_A = 1;
COMP_METHOD_B = 2;

switch(compMethod)
    case(COMP_METHOD_A)
        vX = HybridOrthogonalProjectionOntoConvexSetsA(cProjFun, vY, numIterations, stopThr);
    case(COMP_METHOD_B)
        vX = HybridOrthogonalProjectionOntoConvexSetsB(cProjFun, vY, numIterations, stopThr);
end




end


function [ vX ] = HybridOrthogonalProjectionOntoConvexSetsA( cProjFun, vY, numIterations, stopThr )

numSets     = size(cProjFun, 1);
numElements = size(vY, 1);

vX = vY;
vU = vX;

kk = 0;

for ii = 1:numIterations
    
    vU(:) = vX;
    
    % The loop is as vX changes slowly or not at all with each projection.
    % Hence in order to prevent pre mature stopping one should apply all
    % projections and then generate new vX.
    for jj = 1:numSets
        kk = kk + 1;
        % See Quadratic Optimization of Fixed Points of Non Expensive
        % Mappings in Hilbert Space (Page 12, Equation 20, The note after Equation 21)
        % paramLambdaN = 1 / ((kk + 1) ^ 0.99);
        % See Quadratic Optimization of Fixed Points of Non Expensive
        % Mappings in Hilbert Space (Page 14, Equation 30)
        % Pay attention that Equation 30 allows use of 1 / n.
        paramLambdaN = 1 / (kk + 1);
        % This still require the Fixed Point of the intersection is the
        % same for any cyclic variation of the Sets (Which happens for nay
        % Non Expensive Projection).
        vX(:) = (paramLambdaN * vY) + (1 - paramLambdaN) * cProjFun{jj}(vX);
    end
    
    stopCond = sum(abs(vU - vX)) < stopThr;
    
    if(stopCond)
        break;
    end
end


end


function [ vX ] = HybridOrthogonalProjectionOntoConvexSetsB( cProjFun, vY, numIterations, stopThr )

numSets     = size(cProjFun, 1);
numElements = size(vY, 1);

vX = vY;
vU = vX;
vT = vX;

vW = rand(numSets, 1);
vW = vW / sum(vW);

for ii = 1:numIterations
    
    vU(:) = vX;
    
    % See Quadratic Optimization of Fixed Points of Non Expensive Mappings
    % in Hilbert Space (Page 18, Equation 44)
    vT(:) = 0;
    for jj = 1:numSets
        vT(:) = vT + (vW(jj) * cProjFun{jj}(vX));
    end
    paramLambdaN = 1 / (ii + 1);
    vX(:) = (paramLambdaN * vY) + (1 - paramLambdaN) * vT;
    
    stopCond = sum(abs(vU - vX)) < stopThr;
    
    if(stopCond)
        break;
    end
end


end

