function [ selectedIdx ] = SelectWithoutReplacement( vPdf )
% ----------------------------------------------------------------------------------------------- %
% [ selectedIdx ] = SelectWithoutReplacement( vPdf )
%   Select a number from a given distribution.
% Input:
%   - vProbability  -   Vector for the Probability Distribution Function
%                       (PDF). The i-th element is the probability of the
%                       i-th element to happen.
%                       Structure: Vector (N x 1).
%                       Type: 'Single' / 'Double'.
%                       Range: [0, 1].
% Output:
%   - selectedIdx   -   Selected Index.
%                       The index of element which was raffled.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, ..., N}.
% References
%   1.  a
% Remarks:
%   1.  a
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.000     21/07/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numItems = length(vPdf);

randNum = rand();
ii      = 1;
currSum = sum(vPdf(1:ii));

while((randNum >= currSum) && (ii <= numItems))
    ii      = ii + 1;
    currSum = sum(vPdf(1:ii));
end

selectedIdx = ii;


end

