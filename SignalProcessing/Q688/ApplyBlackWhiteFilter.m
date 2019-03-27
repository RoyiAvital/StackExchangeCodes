function [ mO ] = ApplyBlackWhiteFilter( mI, vCoeffValues )
% ----------------------------------------------------------------------------------------------- %
% [ mO ] = ApplyBlackWhiteFilter( mI, vCoeffValues )
%   Imitates Photoshop's Black & White Adjustment Layer.
% Input:
%   - mI            -   Input Image.
%                       Structure: Image Matrix (Triple Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
%   - vCoeffValues  -   Colors Coefficient Values.
%                       Sets the factor for Reds, Yellows, Greens, Cyans,
%                       Blues and Magentas.
%                       Structure: Vector (6 x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ..., }.
% Output:
%   - mO            -   Output Image.
%                       Structure: Image Matrix (Single Channel)
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% References:
%   1.  What Is the Algorithm Behind Photoshop's “Black and White” Adjustment Layer? - https://dsp.stackexchange.com/questions/688.
%   2.  What Is the Algorithm Behind Photoshop's “Black and White” Adjustment Layer? - https://stackoverflow.com/questions/55185251.
% Remarks:
%   1.  In order to imitate Photoshop the number of coefficients should be
%       6 (Reds, Yellows, Greens, Cyans, Blues, Magentas). The Photoshop
%       range is given in [-200, 300] which should be factored by (1 /
%       100). Basically vCoeffValues = vPhotoshopValues ./ 100.
%   2.  It doesn't implement the 'Tint' option in Photoshop.
% TODO:
%   1.  s
%   Release Notes:
%   -   1.0.000     28/03/2019  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

REDS_IDX        = 1;
YELLOWS_IDX     = 2;
GREENS_IDX      = 3;
CYANS_IDX       = 4;
BLUES_IDX       = 5;
MAGENTAS_IDX    = 6;

numRows = size(mI, 1);
numCols = size(mI, 2);
dataClass = class(mI);

mO = zeros(numRows, numCols, dataClass);

for jj = 1:numCols
    for ii = 1:numRows
        rPx = mI(ii, jj, 1);
        gPx = mI(ii, jj, 2);
        bPx = mI(ii, jj, 3);
        
        grayPx = min(mI(ii, jj, :), [], 3);
        
        rPx = rPx - grayPx;
        gPx = gPx - grayPx;
        bPx = bPx - grayPx;
        
        if(rPx == 0)
            cyanPx      = min(gPx, bPx);
            gPx         = gPx - cyanPx;
            bPx         = bPx - cyanPx;
            
            grayPx = grayPx + (vCoeffValues(GREENS_IDX) * gPx) + (vCoeffValues(CYANS_IDX) * cyanPx) + (vCoeffValues(BLUES_IDX) * bPx);
        elseif(gPx == 0)
            magentaPx   = min(rPx, bPx);
            rPx         = rPx - magentaPx;
            bPx         = bPx - magentaPx;
            
            grayPx = grayPx + (vCoeffValues(REDS_IDX) * rPx) + (vCoeffValues(BLUES_IDX) * bPx) + (vCoeffValues(MAGENTAS_IDX) * magentaPx);
        elseif(bPx == 0)
            yellowPx    = min(rPx, gPx);
            rPx         = rPx - yellowPx;
            gPx         = gPx - yellowPx;
            
            grayPx = grayPx + (vCoeffValues(REDS_IDX) * rPx) + (vCoeffValues(YELLOWS_IDX) * yellowPx) + (vCoeffValues(GREENS_IDX) * gPx);
        end
        mO(ii, jj) = min(max(grayPx, 0), 1);
    end
end


end

