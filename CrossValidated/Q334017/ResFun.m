function [ vResFun ] = ResFun( numMoments, paramMu, paramSigmaSquared, vEmpMoment )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

vResFun = zeros([numMoments, 1]);

for ii = 1:numMoments
    vResFun(ii) = CalcNormalDistributionMoments(paramMu, paramSigmaSquared, ii) - vEmpMoment(ii);
end




end

