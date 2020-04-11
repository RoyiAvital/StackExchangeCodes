
clear();


numRows     = 3;
numVectors  = 6;


%% Load / Generate Data

mW = randn(numRows, numRows);
mW = (mW.' * mW) + (0.00005 + eye(numRows));

mXX = randn(numRows, numVectors); %<! The set of {x}_{i} (Each column)

mX = zeros(numVectors, numRows * numRows);
for ii = 1:numVectors
    mX(ii, :) = kron(mXX(:, ii).', mXX(:, ii).');
end

vY = randn(numVectors, 1);

objVal = 0;

for ii = 1:numVectors
    objVal = objVal + (((mXX(:, ii).' * mW * mXX(:, ii)) - vY(ii)) ^ 2);
end

objVal
sum((mX * mW(:) - vY) .^ 2)

% ii = 8;
% mXX(:, ii).' * mW * mXX(:, ii)
% mX(ii, :) * mW(:)





