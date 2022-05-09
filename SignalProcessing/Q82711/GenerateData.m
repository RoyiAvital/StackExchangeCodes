% StackExchange Signal Processing Q79314
% https://dsp.stackexchange.com/questions/79314
% Image Segmentation Using Deep Learning
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     27/11/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Constants

SIG_TYPE_RAND       = 1;
SIG_TYPE_GAUSSIAN   = 2;


%% Simulation Parameters

% Signal
signalType      = SIG_TYPE_GAUSSIAN;
numSamplesSig   = 201; %<! Make sure it is odd
gaussianStd     = 2;

% Support (Recieved Signal)
numSamples  = 300;

% Noise
minNoiseStd = 0;
maxNoiseStd = 1;
numNoisePts = 50;
numRealizations = 200; %<! Per noise level


%% Generate Data

signalRadius = ceil(4 * gaussianStd);
vSignalSupport = linspace(-signalRadius, signalRadius, numSamplesSig);
vSignal = exp(-(vSignalSupport .^ 2) ./ (2 * gaussianStd));
vSignal = single(vSignal(:));

figure(); plot(vSignalSupport, vSignal);

vNoiseLevel = linspace(minNoiseStd, maxNoiseStd, numNoisePts);
vNoiseLevel = single(vNoiseLevel);

signalRadius    = (numSamplesSig - 1) / 2;
numClasses      = numSamples - numSamplesSig + 1;


mData   = zeros(numSamples, numClasses * numNoisePts * numRealizations, 'single');
vLabels = zeros(numClasses * numNoisePts * numRealizations, 1);

vNoise  = zeros(numSamples, 1, 'single');
vX      = zeros(numSamples, 1, 'single');
sigIdx = 1;

for ii = 1:numClasses
    startIdx = ii;
    endIdx = startIdx + numSamplesSig - 1;
    vX(startIdx:endIdx) = vSignal;
    for jj = 1:numNoisePts
        noiseStd = vNoiseLevel(jj);
        for kk = 1:numRealizations
            vNoise(:) = noiseStd * randn(numSamples, 1, 'single');
            mData(:, sigIdx) = vX + vNoise;
            vLabels(sigIdx) = ii;
            sigIdx = sigIdx + 1;
        end
    end
    vX(startIdx:endIdx) = 0;
end

% ii = 20001; figure(); plot(mData(:, ii)); title(vLabels(ii));

% Verify data by generating it without noise.
% Then:
% [vA, vB] = max(mData, [], 1);
% vC = vB - signalRadius;
% isequal(vC(:), vLabels)



%% Save Data

save('Data', 'vSignal', 'mData', 'vLabels', 'subStreamNumber');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

