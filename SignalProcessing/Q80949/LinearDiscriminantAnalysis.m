function [ mW ] = LinearDiscriminantAnalysis( mX, vY, vP )
% ----------------------------------------------------------------------------------------------- %
%[ mW ] = LinearDiscriminantAnalysis( mX, vY, vP )
% Calculates the projection vectors of Linear Discriminant Analysis (LDA)
% for dimensionality reduction / classification purposes.
% Input:
%   - mX                -   Input Data Matrix.
%                           Input data where a column is a variable and a
%                           row is a sample.
%                           Structure: Matrix (numSamples x numDims).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vY                -   Input Data Labels.
%                           Per sample label.
%                           Structure: Vector (numSamples x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - vP                -   Input Classes Priors.
%                           Per class prior. The sum of priors should sum
%                           to 1.
%                           Structure: Vector (numClasses x 1).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
% Output:
%   - mW                -   LDA Vectors.
%                           Set of vectors of the LDA analysis.
%                           Structure: Matrix (numDims x numDims).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  It is better to have vP scaled by the empirical probability of the
%       classed to the least.
% Known Issues:
%   1.  C
% TODO:
%   1.  D
% Release Notes:
%   -   1.0.000     10/01/2022  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    mX (:, :) {mustBeNumeric, mustBeReal}
    vY (:, 1) {mustBeNumeric, mustBeReal, mustBeInteger}
    vP (:, 1) {mustBeNumeric, mustBeReal} = ones(size(unique(vY), 1), 1, class(mX))
end

[numSamples, numDim] = size(mX);
dataClass = class(mX);

if(size(vY, 1) ~= numSamples)
    error('The number of Labels must match the number of samples');
end

vClasses    = unique(vY);
numClasses  = length(vClasses);
vC          = zeros(numClasses, 1); %<! Num samples per class

if(size(vP, 1) ~= numClasses)
    error('The number of Priors must match the number of classes');
end

for ii = 1:numClasses
    vC(ii) = sum(vY == vClasses(ii));
end

% Calculate the Mean Vectors
mM = zeros(numClasses, numDim, dataClass);

for ii = 1:numClasses
    mM(ii, :) = mean(mX(vY == vClasses(ii), :), 1);
end

% Calculate the Intra Class Covariance
% mSw = zeros(numDim, numDim, numClasses, dataClass);
% for ii = 1:numSamples
%     classIdx            = find(vY(ii) == vClasses);
%     mSw(:, :, classIdx) = mSw(:, :, classIdx) + CovProduct(mX(ii, :), mM(classIdx, :));
% end
% mSw = sum(mSw, 3); %<! Not neeeded

mSw = zeros(numDim, numDim, dataClass);
for ii = 1:numSamples
    classIdx  = find(vY(ii) == vClasses); %<! Use find to support any way to designate a class
    mSw(:, :) = mSw(:, :) + (vP(classIdx) * CovProduct(mX(ii, :), mM(classIdx, :)));
end

% Calculate the Inter Class Covariance
vM = mean(mX, 1);
mSb = zeros(numDim, numDim, dataClass);

for ii = 1:numClasses
    mSb = mSb + (vC(ii) * CovProduct(mM(ii, :), vM));
end

% [mV, vD] = eig(mSw \ mSb, 'vector');
[mV, vD] = eig(mSb, mSw, 'vector');

% The function eig() isn't guaranteed to return in a specific order
[~, vSortIdx] = sort(vD, 'descend');

mW = mV(:, vSortIdx);

    
end


function [ mC ] = CovProduct( vA, vB )

vC = vA(:) - vB(:);
mC = vC * vC.';


end
