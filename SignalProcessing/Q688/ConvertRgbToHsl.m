function [ mHslImage ] = ConvertRgbToHsl( mRgbImage )
% ----------------------------------------------------------------------------------------------- %
%[ mHslImage ] = ConvertRgbToHsl( mRgbImage )
% Converts the Input RGB Image to HSL Color Model.
% Input:
%   - mRgbImage         -   Input Image.
%                           Structure: Image Matrix (3 Channels).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
% Output:
%   - mHslImage         -   HSL Image.
%                           Hue [0, 360) [Deg] -> [0, 1).
%                           S in [0, 1].
%                           L in [0, 1].
%                           Structure: Image Matrix (3 Channels).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
% References
%   1.  HSL and HSV (Wikipedia) - https://en.wikipedia.org/wiki/HSL_and_HSV.
% Remarks:
%   1.  This implementation uses Alph, Beta, H2 and C2 formulation of
%       Wikipedia. This is the Circle form instead of the Hexagon form.
%   2.  Pure colors (Red, Green, Blue) has L of 0.5.
% TODO:
%   1.  A
% Release Notes:
%   -   1.0.000     22/12/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

OFF = 0;
ON  = 1;

numRows = size(mRgbImage, 1);
numCols = size(mRgbImage, 2);

mRgbImage = reshape(mRgbImage, numRows * numCols, 3);

vMax = max(mRgbImage, [], 2);
vMin = min(mRgbImage, [], 2);

vC = vMax - vMin;

% Using Wikipedia Alpha & Beta with H2 and C2
% atan2() Returns values on the range [-pi, pi]
vH = atan2((sqrt(3) / 2) * (mRgbImage(:, 2) - mRgbImage(:, 3)), 0.5 * ((2 * mRgbImage(:, 1)) - mRgbImage(:, 2) - mRgbImage(:, 3)));
vH = mod(vH + (2 * pi), 2 * pi) / (2 * pi); %<! Normalizing into [0, 1) range

vL = (vMax + vMin) / 2;

vS = vC ./ (1 - abs((2 * vL) - 1));
vS(vC == 0) = 0;

mHslImage = reshape([vH, vS, vL], numRows, numCols, 3);


end

