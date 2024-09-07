% StackExchange Signal Processing Q81138
% https://dsp.stackexchange.com/questions/81138
% Using Least Mean Square (LMS) Filter for Beamforming on Linear Array in
% Julia
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     09/04/2022
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
numElements = 20;
distElmFctr = 1; %<! Factor of el

% Signals
timeInterval    = 5;
sigFreq         = 1e3; %<! [Hz]
samplingFreq    = 100 * sigFreq; %<! [Hz]

% Target
targetAmp       = 1;
targetAzimuth   = 21; %<! [Deg]

% Interference
vIntAmp     = [0; 0; 0];
vIntAzimuth = [20; -30; 50]; %<! [Deg]

% Noise
noiseAmp = 0.1; %<! Standard deviation

% Initial Weight Vector
vW = zeros(numElements, 1);

numAzimuths = 360;


%% Generate / Load Data

vT = linspace(0, timeInterval, (samplingFreq * timeInterval) + 1);
vT = vT(:);
vT(end) = [];

numSamples = length(vT);

waveLen = SPEED_OF_LIGHT_M_S / sigFreq;
distElm = distElmFctr * (waveLen / 2);


vR = sin(2 * pi * sigFreq * vT); %<! Reference signal

vP = CalcPhaseVec(targetAzimuth, distElm, sigFreq, numElements);

mX = GenSensorsSignal(vR, vP);

figure();
plot(mX);



%% LMS
% vW(1) = 1;
% vW(2) = 0.1;

vWW = LmsFilter(vW, mX, vR, numSamples, 5e-4, OFF);


%% Analysis

recError = norm(mX * vWW - vR);
disp(['Reconstruction Error:', num2str(recError)]);

vH = CalcUlaPattern(vWW, distElm, sigFreq, numElements, numAzimuths);

vA = linspace(0, 2 * pi, numAzimuths + 1);
vA = vA(:);
vA(end) = [];

figure();
polarplot(vA, abs(vH));


%% Auxiliary Functions

function [ vP ] = CalcPhaseVec( aziAngle, distElm, sigFreq, numElements )

SPEED_OF_LIGHT_M_S = 3e9;

vP = ((distElm * sind(aziAngle)) / SPEED_OF_LIGHT_M_S) * sigFreq * (0:(numElements - 1)); %<! Must be a row vector

end

function [ mX ] = GenSensorsSignal( vS, vP )

% Adding delay to each signal by adding a phase to the analytic signal
vA = exp(-2j * pi * vP); %<! vP assumed to be a row vector
mX = real(hilbert(vS) .* vA);

end

function [ vH ] = CalcUlaPattern( vW, distElm, sigFreq, numElements, numAzimuths )

SPEED_OF_LIGHT_M_S = 3e9;

vA = linspace(0, 360, numAzimuths + 1);
vA = vA(:);
vA(end) = [];

% vW = hilbert(vW);

vH = zeros(numAzimuths, 1, 'like', 1i);

for ii = 1:numElements
    vH = vH + vW(ii) * exp(-2j * pi * ((distElm * sind(vA)) / SPEED_OF_LIGHT_M_S) * sigFreq * (ii - 1));
end

end



%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

