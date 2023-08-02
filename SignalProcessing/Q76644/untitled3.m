
clear();
close('all');

% Signal Parameters
numSamples      = 100;
samplingFreq    = 1; %<! The CRLB is for Normalized Frequency

% Sine Signal Parameters (Non integeres divsiors of N requires much more realizations).
sineFreq    = 0.25; %<! Do for [0.05, 0.10, 0.25] For no integer use 0.37.
sineAmp     = 10; %<! High value to allow high SNR
angFreq     = 2 * pi * (sineFreq / samplingFreq);
sinePhase   = 2 * pi * rand(1, 1);


vS = sineAmp * sin((angFreq * (0:(numSamples - 1))) + sinePhase);
vSS = fft(vS);

PlotDft(vS, samplingFreq);

