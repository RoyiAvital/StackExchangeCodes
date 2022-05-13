% StackExchange Signal Processing Q82918
% https://dsp.stackexchange.com/questions/82918
% MUSIC Algorithm for Direction of Arrival (DOA) in Acoustic Signals
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     09/05/2022
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Simulation Constants

SPEED_OF_LIGHT_M_S = 3e9;


%% Simulation Parameters

% Array
numElements = 80;
distElmFctr = 1; %<! Factor of Lambda / 2

% Signals
timeInterval    = 5;
vSigAmp         = [1; 1; 1];
sigFreq         = 1e1; %<! [Hz]
vSigPhase       = [0; 29; 61]; %<! [Deg]
vSigAzimuth     = [-60; 0; 40];
samplingFreq    = 10 * sigFreq; %<! [Hz]

% Noise
noiseAmp = 0.01; %<! Standard deviation

% MUSIC
numGridPts = 361; %<! Grid of Angles


%% Generate / Load Data

numSig = length(vSigAzimuth);

vT = linspace(0, timeInterval, (samplingFreq * timeInterval) + 1);
vT = vT(:);
vT(end) = [];

numSamples = length(vT);

waveLen = SPEED_OF_LIGHT_M_S / sigFreq;
distElm = distElmFctr * (waveLen / 2);

mR = zeros(numSamples, numSig); %<! Signals

for ii = 1:numSig
    sigPhase = vSigPhase(ii) / (2 * pi); %<! [Radians]
    mR(:, ii) = vSigAmp(ii) * sin(2 * pi * sigFreq * vT + sigPhase);
end


%% MUSIC Algorithm

mA = zeros(numElements, numSig, 'like', 1i); %<! Steering Matrix

for ii = 1:numSig
    mA(:, ii) = exp(-2j * pi * CalcPhaseVec(vSigAzimuth(ii), distElm, sigFreq, numElements));
end

mX = hilbert(mR) * mA'; %<! MATLAB generates the analytic signal itself
% mX = mR * mA'; %<! Without Hilbert Transform
mX = mX + (noiseAmp * (randn(numSamples, numElements) + 1j * randn(numSamples, numElements)));

mC = cov(mX); %<! Covariance Matrix

[mEigVec, vEigVal] = eig(mC, 'vector'); %<! Eigen Vectors (Spanning space of Signal / Noise)
[vEigVal, vIdx] = sort(vEigVal, 'ascend'); %<! Signal values at the end (Stronger)
mEigVec = mEigVec(:, vIdx);

mV = mEigVec(:, 1:(numElements - numSig)); %<! Spanning the Noise (Assuming Signal Eigen Values are larger)

vTheta = linspace(-90, 90, numGridPts); %<! Grid for Estimation
vS = zeros(numElements, 1);
vM = zeros(numGridPts, 1);

% for ii = 1:length(vTheta)
%     % vS(:) = exp(-2j * pi * 2 * (0:(numElements - 1))' * sind(vTheta(ii)));
%     vS(:) = exp(-2j * pi * CalcPhaseVec(vTheta(ii), distElm, sigFreq, numElements));
%     vM(ii) = abs(1 / (vS' * (mV * mV') * vS));
% end

% Optimized Loop

mVV = mV * mV';
% [mU, mD] = ldl(mVV, 'upper');
% mD = max(mD, 0); %<! mVV must be PSD Matrix -> Lowest Eig must be zero
% mU = sqrt(mD) * mU; %<! Pseudo Cholesky
mU = sqrtm(mVV);

for ii = 1:numGridPts
    vS(:) = exp(-2j * pi * CalcPhaseVec(vTheta(ii), distElm, sigFreq, numElements));
    vS(:) = mU * vS;
    % vM(ii) = abs(1 / (vS' * vS));
    vM(ii) = 1 / (vS' * vS); %<! From the decomposition and structure should be real!
end


[vPeakVal, vPeakIdx] = findpeaks(vM, 'SortStr', 'descend');


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(vTheta, 10 * log10(abs(vM)));
set(hLineObj, 'LineWidth', lineWidthNormal);

set(get(hAxes, 'Title'), 'String', {['MUSIC Pseudo Spectrum']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Spatial Angle [Deg]']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Value [dB]']}, ...
    'FontSize', fontSizeAxis);

hLineObj = plot(vTheta(vPeakIdx(1:numSig)), 10 * log10(abs(vPeakVal(1:numSig))));
set(hLineObj, 'LineStyle', 'none', 'LineWidth', lineWidthNormal, 'Marker', 'x', 'MarkerSize', markerSizeLarge, 'Color', 'r');
for ii = 1:numSig
    hLineObj = xline(vTheta(vPeakIdx(ii)), '-', {['Angle: ', num2str(vTheta(vPeakIdx(ii))), ' [Deg]']});
    set(hLineObj, 'LineStyle', ':', 'LineWidth', lineWidthThin, 'Color', 'r');
end

% hLegend = ClickableLegend({['Ground Truth'], ['Input Noisy Samples'], ['TV Estimation']});

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxilizary Functions

function [ vP ] = CalcPhaseVec( aziAngle, distElm, sigFreq, numElements )

SPEED_OF_LIGHT_M_S = 3e9;

vP = ((distElm * sind(aziAngle)) / SPEED_OF_LIGHT_M_S) * sigFreq * (0:(numElements - 1)); %<! Must be a row vector

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

