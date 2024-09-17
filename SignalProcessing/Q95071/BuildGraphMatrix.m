function [ mW ] = BuildGraphMatrix( mI, hV, hW, winRadius )
% ----------------------------------------------------------------------------------------------- %
% [ mW ] = BuildGraphMatrix( mI, hW, winRadius )
%   Build a graph of `LxL` neighborhood using the weights function.
% Input:
%   - mI                -   Input Image.
%                           Structure: Matrix (numRows x numCols).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - hV                -   Connectivity (Validation) Function.
%                           Given the indices of the 2 pixels sets whether
%                           the graph connects them.
%                           Structure: Scalar.
%                           Type: Function Handler.
%                           Range: NA.
%   - hW                -   Weights Function.
%                           Given 2 values of pixels calculates the weight
%                           of their edge.
%                           Structure: Scalar.
%                           Type: Function Handler.
%                           Range: NA.
%   - winRadius        -   Window Radius.
%                           The local radius.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {1, 2, ...}.
% Output:
%   - mW                -   Graph Matrix.
%                           Structure: Sparse Matrix (numPx x numPx).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References:
%   1.  A
% Remarks:
%   1.  MATLAB removes implicit zeros. So if `hW()` returns a zero value
%       for some case it will not be included in the output of `find()`.
% TODO:
%   1.  Add the indices to the input of `hW()`.
%   Release Notes:
%   -   1.0.000     16/09/2024  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    mI (:, :) {mustBeNumeric, mustBeNonnegative, mustBeFloat}
    hV (1, 1) {mustBeA(hV, 'function_handle')}
    hW (1, 1) {mustBeA(hW, 'function_handle')}
    winRadius (1, 1) {mustBePositive, mustBeInteger}
end

arguments(Output)
    mW (:, :) {mustBeSparse, mustBeFloat}
end

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx   = numRows * numCols;
winLen  = (2 * winRadius) + 1;

% Number of edges (Ceiled estimation as on edges there are less)
vI = ones(winLen * winLen * numPx, 1);  %<! Must be valid index
vJ = ones(winLen * winLen * numPx, 1);  %<! Must be valid index
vV = zeros(winLen * winLen * numPx, 1); %<! Add zero value

elmIdx = 0;
refPxIdx = 0;
for jj = 1:numCols
    for ii = 1:numRows
        refPxIdx = refPxIdx + 1;
        for nn = -winRadius:winRadius
            for mm = -winRadius:winRadius
                if (((ii + mm) > 0) && ((ii + mm) <= numRows) && ((jj + nn) > 0) && ((jj + nn) <= numCols))
                    % Pair is within neighborhood
                    isValid = hV(ii, jj, mm, nn); %<! Connectivity
                    if (isValid)
                        weightVal   = hW(mI(ii, jj), mI(ii + mm, jj + nn)); %<! Value
                        elmIdx      = elmIdx + 1;
                        vI(elmIdx)  = refPxIdx;
                        vJ(elmIdx)  = refPxIdx + (nn * numRows) + mm;
                        vV(elmIdx)  = weightVal;
                    end
                end
            end
        end
    end
end

mW = sparse(vI, vJ, vV, numPx, numPx);


end

