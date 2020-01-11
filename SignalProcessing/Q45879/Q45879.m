% StackExchange Signal Processing Q45879
% https://dsp.stackexchange.com/questions/45879
% Estimate Filter Coefficients from the Result of Linear Convolution with a
% Known Signal
% References:
%   1.  aa
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     11/01/2020
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Simulation Parameters

numTaps     = 4; %<! Filter Coefficients
numSamples  = 25; %<! Data Samples

cMethodNames = {['Least Squares'], ['Levinson Recursion'], ['Circular Convolution']};
methodIdx = 0;


%% Generate Data

vH = randn(numTaps, 1);
vX = randn(numSamples, 1);
vY = conv(vX, vH, 'full');


%% Solution by Linear Algebra

methodIdx = methodIdx + 1;

mX = GenerateToeplitzConvMatrix(vX, numTaps, CONVOLUTION_SHAPE_FULL);
vHH = mX \ vY;
maxAbsDev = max(abs(vHH - vH));

disp([' ']);
disp([cMethodNames{methodIdx}, ' Solution Summary']);
disp(['The Maximum Absolute Deviation Is Given By - ', num2str(maxAbsDev)]);
disp([' ']);


%% Solution by Levinson Recursion

methodIdx = methodIdx + 1;

mX = GenerateToeplitzConvMatrix(vX, numTaps, CONVOLUTION_SHAPE_FULL);
mH = zeros(numTaps, size(mX, 1) / numTaps);
for ii = 1:size(mX, 1) / numTaps
    firstIdx = (numTaps * (ii - 1)) + 1;
    lastIdx = numTaps * ii;
    mH(:, ii) = LevinsonRecursion(mX(firstIdx:lastIdx, :), vY(firstIdx:lastIdx));
end
vHH = mean(mH, 2);
maxAbsDev = max(abs(vHH - vH));

disp([' ']);
disp([cMethodNames{methodIdx}, ' Solution Summary']);
disp(['The Maximum Absolute Deviation Is Given By - ', num2str(maxAbsDev)]);
disp([' ']);


%% Solution by Circular Convolution

methodIdx = methodIdx + 1;

vYBar = sum(buffer(vY, numSamples), 2);
vH = ifft(fft(vYBar) ./ fft(vX), 'symmetric');
vH = vH(1:numTaps);
maxAbsDev = max(abs(vHH - vH));

disp([' ']);
disp([cMethodNames{methodIdx}, ' Solution Summary']);
disp(['The Maximum Absolute Deviation Is Given By - ', num2str(maxAbsDev)]);
disp([' ']);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

