function [ vY ] = BilateralFilter1D( vX, kernelRadius, timeStd, valueStd )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = ImageConvFrequencyDomain( mI, mH, convShape )
% Applies Image Convolution in the Frequency Domain.
% Input:
%   - vX                -   Input Signal.
%                           Structure: Vector (numSamples x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - kernelRadius      -   Kernel Radius.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}
%   - timeStd           -   Time Axis Std.
%                           The Standard Deviation of the exponential
%                           weights along the time axis (The axis of the
%                           support o the signal).
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - valueStd          -   Value Std.
%                           The Standard Deviation of the exponential
%                           weights of the samples value.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - mI                -   Output Image.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  The higher the values of `timeStd` or `valueStd` the stronger the
%       smoothing.
%   2.  For high accuracy make sure that `kernelRadius >= ceil(4 * timeStd)`.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     18/07/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    vX (:, 1) {mustBeNumeric, mustBeReal}
    kernelRadius (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = 3
    timeStd (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = 1
    valueStd (1, 1) {mustBeNumeric, mustBeReal, mustBePositive} = -1
end

if(valueStd == -1)
    valueStd = std(vX) / 10;
end

numSamples  = size(vX, 1);
winLen      = (2 * kernelRadius) + 1;
dataClass   = class(vX);

timeFactor  = 2 * timeStd * timeStd;
valueFactor = 2 * valueStd * valueStd;

vY = zeros(numSamples, 1, dataClass);

vWT = exp(-((-kernelRadius:kernelRadius) .^ 2 / timeFactor));
vWT = vWT(:);

firstIdx = kernelRadius + 1;
shiftLeft = 0;
for ii = 1:kernelRadius
    valX    = vX(ii);
    vV      = vX((ii - shiftLeft):(ii + kernelRadius)); %<! Window
    vWV     = exp(-(((vV - valX) .^ 2) / valueFactor)); %<! Weights for value range
    vW      = vWV .* vWT(firstIdx:end);
    vW      = vW ./ sum(vW);
    vY(ii)  = sum(vW .* vV);
    
    firstIdx    = firstIdx - 1;
    shiftLeft   = shiftLeft + 1;
end

vV  = zeros(winLen, 1, dataClass);
vWV = zeros(winLen, 1, dataClass);
vW  = zeros(winLen, 1, dataClass);

for ii = (kernelRadius + 1):(numSamples - kernelRadius)
    valX    = vX(ii);
    vV(:)   = vX((ii - kernelRadius):(ii + kernelRadius)); %<! Window
    vWV(:)  = exp(-(((vV - valX) .^ 2) / valueFactor)); %<! Weights for value range
    vW(:)   = vWV .* vWT;
    vW(:)   = vW ./ sum(vW);
    vY(ii)  = sum(vW .* vV);
end

lastIdx     = winLen - 1;
shiftRight  = kernelRadius - 1;
for ii = (numSamples - kernelRadius + 1):numSamples
    valX    = vX(ii);
    vV      = vX((ii - kernelRadius):(ii + shiftRight)); %<! Window
    vWV     = exp(-(((vV - valX) .^ 2) / valueFactor)); %<! Weights for value range
    vW      = vWV .* vWT(1:lastIdx);
    vW      = vW ./ sum(vW);
    vY(ii)  = sum(vW .* vV);
    
    lastIdx     = lastIdx - 1;
    shiftRight  = shiftRight - 1;
end


end

