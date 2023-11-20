function [ vX, mX ] = ChambollePock( vX, mX, vP, hGradP, hProjP, hStepX, stepSize, numIterations )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

for ii = 2:numIterations    
    % Projected Sub Gradient Method on Dual (vP)
    vP(:)   = vP - (stepSize * hGradP(vP, vX));
    vP(:)   = hProjP(vP);
    
    vX(:) = hStepX(vX, vP);
    
    mX(:, ii) = vX;
end


end

