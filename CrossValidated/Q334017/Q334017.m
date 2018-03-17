% Cross Validated Q334017
% https://stats.stackexchange.com/questions/334017
% Two Parameter Method of Moments Estimation
% References:
%   1.  Method of Moments - https://en.wikipedia.org/wiki/Method_of_moments_(statistics).
%   2.  Normal Distribution Moments - https://en.wikipedia.org/wiki/Normal_distribution#Moments.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     17/03/2018
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numSamples      = 1000;
paramMu         = 2;
paramSigmaSq    = 3; %<! Sigma Squared

numMoments = 3;


%% Generate Data

vX = paramMu + (sqrt(paramSigmaSq) * randn([numSamples, 1]));
vEmpMoment = zeros([numMoments, 1]);


%% Generating Empirical Moments

for ii = 1:numMoments
    vEmpMoment(ii) = mean(vX .^ ii);
end

hResFun = @(vTheta) ResFun(numMoments, vTheta(1), vTheta(2), vEmpMoment);
vTheta = [0, 1];

sSolverOptions = optimoptions('lsqnonlin', 'Algorithm', 'trust-region-reflective', 'FunctionTolerance', 1e-9, 'StepTolerance', 1e-9, 'FiniteDifferenceType', 'central', 'Display', 'iter');

vTheta = lsqnonlin(hResFun, vTheta, [], [], sSolverOptions);


%% Analysis

disp(['']);
disp(['Results Analysis']);
disp(['Estimated Mean - ', num2str(vTheta(1)), ', Ground Truth Mean - ', num2str(paramMu)]);
disp(['Estimated Variance - ', num2str(vTheta(2)), ', Ground Truth Variance - ', num2str(paramSigmaSq)]);
disp(['']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

