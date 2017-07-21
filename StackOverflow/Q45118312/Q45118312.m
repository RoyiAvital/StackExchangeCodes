% Stack Overflow Q45118312
% https://stackoverflow.com/questions/45118312
% Estimate Poisson PDF Parameters Using Curve Fitting in MATLAB
% References:
%   1.  Poisson Distribution - https://en.wikipedia.org/wiki/Poisson_distribution.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     21/07/2017
%   *   First release.


%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Simulation Parameters

numTests            = 50;
numSamples          = 1000;
paramLambdaBound    = 10;
epsVal              = 1e-6;

hPoissonPmf = @(paramLambda, vParamK) ((paramLambda .^ vParamK) * exp(-paramLambda)) ./ factorial(vParamK);


for ii = 1:ceil(1000 * paramLambdaBound)
    if(hPoissonPmf(paramLambdaBound, ii) <= epsVal)
        break;
    end
end

vValGrid = [0:ii];
vValGrid = vValGrid(:);

vParamLambda    = zeros([numTests, 1]);
vParamLambdaMl  = zeros([numTests, 1]); %<! Maximum Likelihood
vParamLambdaCf  = zeros([numTests, 1]); %<! Curve Fitting


%% Generate Data and Samples

for ii = 1:numTests
    
    paramLambda = paramLambdaBound * rand([1, 1]);
    
    vDataSamples    = poissrnd(paramLambda, [numSamples, 1]);
    vDataHist       = histcounts(vDataSamples, [vValGrid - 0.5; vValGrid(end) + 0.5]) / numSamples;
    vDataHist       = vDataHist(:);
    
    vParamLambda(ii)    = paramLambda;
    vParamLambdaMl(ii)   = mean(vDataSamples); %<! Maximum Likelihood
    vParamLambdaCf(ii)   = lsqcurvefit(hPoissonPmf, 2, vValGrid, vDataHist, 0, inf); %<! Curve Fitting
end

hFigure = figure('Position', figPosLarge);
hAxes   = axes();
hLineSeries = plot([1:numTests], [vParamLambda, vParamLambdaMl, vParamLambdaCf]);
set(hLineSeries, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', markerSizeNormal);
set(get(hAxes, 'Title'), 'String', {['Poisson Paraneter Estimation - Maximum Likelihood vs. PDF Curve Fitting'], ...
    ['ML RMSE - ', num2str(norm(vParamLambdaMl - vParamLambda)), ', Curve Fitting MSE - ', num2str(norm(vParamLambdaCf - vParamLambda)), ', Num Samples - ', num2str(numSamples)]}, ...
    'Fontsize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Trial Number']}, ...
    'Fontsize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['\lambda Value']}, ...
    'Fontsize', fontSizeAxis);
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
hLegend = ClickableLegend({['\lambda'], ['\lambda - Maximum Likelihood'], ['\lambda - PDF Curve Fitting']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

