function [ valRoot ] = FindRootBiSection( hInputFun, lowerBound, upperBound, stopThr )
% ----------------------------------------------------------------------------------------------- %
%[ valRoot ] = FindRootBiSection( hInputFun, lowerBound, upperBound, stopThr )
% Finds a root of the input function with the given segment [lowerBound,
% upperBound].
% Input:
%   - hInputFun         -   Input Function Handler.
%                           Calculates the function value at the Input
%                           Point.
%                           Structure: Function Handler.
%                           Type: Handler.
%                           Range: NA.
%   - lowerBound        -   Lower Bound.
%                           Lower bound value of the segment.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - upperBound        -   Upper Bound.
%                           Upper bound value of the segment.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - stopThr           -   Stopping Threshold.
%                           Sets the threshold for the distance between the
%                           2 boundaries of the segment before termination.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
% Output:
%   - valRoot           -   Root Value.
%                           The value of a root of the function within the
%                           segment. It is within a `stopThr` distance from
%                           the actual root.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  The values of the function at the boundaries of the segments must
%       have different signs or at least one of them is a root.
% TODO:
%   1.  A
% Release Notes:
%   -   1.0.000     06/01/2020  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

lowerBoundVal = hInputFun(lowerBound);
upperBoundVal = hInputFun(upperBound);

if(abs(lowerBoundVal) <= stopThr)
    valRoot = lowerBound;
    return;
end

if(abs(upperBoundVal) <= stopThr)
    valRoot = upperBound;
    return;
end

if(lowerBoundVal * upperBoundVal > 0)
    error(['The input function must have different sign at the boundaries of the segment']);
end

while(abs(upperBound - lowerBound) > stopThr)
    valRoot = (lowerBound + upperBound) / 2;
    if((hInputFun(lowerBound) * hInputFun(valRoot)) > 0)
        lowerBound = valRoot;
    else
        upperBound = valRoot;
    end
end


end

