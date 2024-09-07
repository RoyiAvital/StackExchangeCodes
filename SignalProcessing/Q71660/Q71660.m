% StackExchange Signal Processing Q71660
% https://dsp.stackexchange.com/questions/71660
% Deconvolution with Noisy Measurement of the Model Coefficients
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     01/04/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 5223;
subStreamNumberDefault = 0;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

CONVOLUTION_SHAPE_FULL         = 1;
CONVOLUTION_SHAPE_SAME         = 2;
CONVOLUTION_SHAPE_VALID        = 3;


%% Parameters

vH          = [0.05; 0.65; 0.1; 0.15; 0.05];
numSamples  = 600;

noiseStd = 0.75;


%% Generate / Load Data

vX = rand(numSamples, 1); %<! Input Data
vY = conv(vH, vX, 'full'); %<! Reference

% Noisy data
vHH = vH + (noiseStd * randn(length(vH), 1));
vYY = vY + (noiseStd * randn(length(vY), 1));

mH  = full(CreateConvMtx1D(vH, length(vX), CONVOLUTION_SHAPE_FULL));
mHH = full(CreateConvMtx1D(vHH, length(vX), CONVOLUTION_SHAPE_FULL));


%% Analysis

vXLS    = mHH \ vYY; %<! Least Squares Solution
vXTLS   = TlsRegression(mHH, vYY); %<! Total Least squares Solution

mseLS  = mean((vXLS - vX) .^ 2);
mseTLS = mean((vXTLS - vX) .^ 2);


%% Display Results

disp(['The Least Squares solution MSE vs. the ground truth      : ', num2str(mseLS)]);
disp(['The Total Least Squares solution MSE vs. the ground truth: ', num2str(mseTLS)]);


%% Auxiliary Functions

function [ vX ] = TlsRegression( mH, vY )

numCols = size(mH, 2);

[~, ~, mV] = svd([mH, vY]);
vX = -mV(1:numCols, numCols + 1) / mV(numCols + 1, numCols + 1);
[~, vD, ~] = svd([mH, vY], 'vector');
disp(vD(end));

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

