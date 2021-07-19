function [ hScatterObj ] = ScatterSparse( mS, varargin )
% ----------------------------------------------------------------------------------------------- %
% [ hScatterObj ] = ScatterSparse( mS, varargin )
% Visualize sparsity pattern of matrix. Comparable to `spy()` with all the
% controls given by `scatter()`.
% Input:
%   - mS                -   Input Sparse Matrix.
%                           Structure: Matrix (numRows x numCols).
%                           Type: 'Single' / 'Double' (Sprase).
%                           Range: (-inf, inf).
%   - varargin          -   Scatter Plot Parameters.
%                           Set of parameters accepeted by `scatter()`.
%                           Structure: NA.
%                           Type: NA.
%                           Range: NA.
% Output:
%   - hScatterObj       -   Scatter Plot Object.
%                           Structure: Scalar.
%                           Type: Handler / Object.
%                           Range: NA.
% References:
%   1.  A
% Remarks:
%   1.  Supports any functionality by `scatter()` plot.
% TODO:
%   1.  
%   Release Notes:
%   -   1.0.000     19/07/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    mS (:, :) {mustBeNumeric, mustBeReal, mustBeSparse}
end
arguments (Repeating)
    varargin
end

[vI, vJ, ~] = find(mS);
hScatterObj = scatter(vJ, vI, varargin{:});


end


function [ ] = mustBeSparse( mS )
    
if(~issparse(mS))
    eid = 'mustBeSparse:mSMustBeSparse';
    msg = 'The input matrix must be sparse.';
    throwAsCaller(MException(eid, msg));
end
    
    
end

