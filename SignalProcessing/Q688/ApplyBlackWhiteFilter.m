function [ mO ] = ApplyBlackWhiteFilter( mI, vCoeffValues )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = ApplyBlackWhiteFilter( mI, vCoeffValues )
%   Imitates Photoshop's Black & White Adjustment Layer.
% Input:
%   - mI            -   Input Image.
%                       Structure: Image Matrix (Triple Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - boxRadius     -   Box Radius.
%                       The radius of the box neighborhood for the
%                       filtering process.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ..., }.
%   - borderType    -   Border Type.
%                       Sets the border extension mode (Cconstant,
%                       Circular, Replicate, Symmetric).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3 4}.
%   - borderValue   -   Border Value.
%                       Sets the border value in case Border Type is
%                       Constant.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% Output:
%   - mO            -   Output Image.
%                       Structure: Image Matrix (Single Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% References:
%   1.  What Is the Algorithm Behind Photoshop's “Black and White” Adjustment Layer? - https://dsp.stackexchange.com/questions/688.
% Remarks:
%   1.  In order to imitate Photoshop the number of coefficients should be
%       6 (Reds, Yellows, Greens, Cyans, Blues, Magentas). The Photoshop
%       range is given in [-200, 300] which should be linearly mapped into
%       [-5, 5] where 50 is mapped to 0. Basicall vCoeffValues = (vPhotoshopValues - 50) ./ 50.
%   2.  It doesn't implement the 'Tint' option in Photoshop.
% TODO:
%   1.  s
%   Release Notes:
%   -   1.0.000     22/12/2018  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

numRows = size(mI, 1);
numCols = size(mI, 2);
dataClass = class(mI);

numCoeff    = size(vCoeffValues, 1);
hueRadius   = 1 / numCoeff;
vHueVal     = [0:(numCoeff - 1)] * hueRadius;

mHsl = ConvertRgbToHsl(mI);
mO = zeros(numRows, numCols, dataClass);

vCoeffValues = numCoeff * vCoeffValues;

for jj = 1:numCols
    for ii = 1:numRows
        hueVal = mHsl(ii, jj, 1);
        lumCoeff = 0;
        
        % For kk = 1 we're dealing with circular distance
        diffVal     = min(abs(vHueVal(1) - hueVal), abs(1 - hueVal));
        lumCoeff    = lumCoeff + (vCoeffValues(1) * max(0, hueRadius - diffVal));
        for kk = 2:numCoeff
            lumCoeff = lumCoeff + (vCoeffValues(kk) * max(0, hueRadius - abs(vHueVal(kk) - hueVal)));
        end
        
        mO(ii, jj) = mHsl(ii, jj, 3) * (1 + lumCoeff);
    end
end


end

