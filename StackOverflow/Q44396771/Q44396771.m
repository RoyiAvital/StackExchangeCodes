% StackOverflow Q44396771
% Simulation of Bernouli Trial, Expnonential Process and Poisson Process
% for simulating Premium and Claims of Insuarance Company.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     06/06/2017
%   *   First release.
%

%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';


%% Setting Parameters

numSimulations = 100;

vParamU = [100, 500]; %<! Starting reserve
vParamP = [0.1, 0.2]; %<! Collection Probability - Bernoulli Trial
vParamG = [0.5, 1, 2]; %<! Claim Rate - Poisson Process
vParamC = [50, 100, 200]; %<! Claim Value - Exponential Process
vParamH = [10, 100]; %<! Time Units (Samples)

tBankruptcyFlag = zeros(numSimulations, length(vParamC), length(vParamG), length(vParamP), length(vParamU));


for ii = 1:length(vParamU)
    paramU = vParamU(ii);
    for jj = 1:length(vParamP)
        paramP = vParamP(jj);
        for kk = 1:length(vParamG)
            paramG = vParamG(kk);
            for ll = 1:length(vParamC)
                paramC = vParamC(ll);
                for mm = 1:numSimulations
                    for nn = 1:length(vParamH)
                        paramH = vParamH(nn);
                        
                        currValue = paramU;
                        
                        for oo = 1:paramH
                            
                            collectedSum    = (rand(1) <= paramP) * paramP;
                            numClaims       = poissrnd(paramG, [1, 1]);
                            claimedSum      = sum(exprnd(paramC, [numClaims, 1]));
                            
                            currValue = currValue + collectedSum - claimedSum;
                        end
                        
                        tBankruptcyFlag(mm, ll, kk, jj, ii) = currValue < 0;
                        
                    end
                end
                
                bankruptcyProb = mean(tBankruptcyFlag(:, ll, kk, jj, ii));
                disp(['Probability of Bankruptcy for C = ', num2str(paramC), ', Lambda = ', num2str(paramG), ', P = ', num2str(paramP), ', U = ', num2str(paramP), ' is ', num2str(bankruptcyProb)]);
                disp(['Number of Simulations - ', num2str(numSimulations)]);
                disp(['Number of Time Units - ', num2str(paramH)]);
                disp(['']);
                
            end
        end
    end
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

