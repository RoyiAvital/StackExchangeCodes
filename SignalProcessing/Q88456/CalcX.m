
clear();
close('all');

mA = readmatrix("mA.csv");
mR = readmatrix("mR.csv");
mY = readmatrix("mY.csv");

paramLambda = 1;
tic()
mW = kron(mA.', mR);

mWW = mW.' * mW;
mWWI = mWW + (paramLambda * eye(size(mWW)));

% vX = mW \ mY(:);
vX = mWWI \ (mW.' * mY(:));

% vX = ridge(mY(:), mW, paramLambda, 0);
% vX = lasso(mW, mY(:), 'Lambda', paramLambda);

% vX = lsqnonneg(mW, mY(:));
% vX = lsqnonneg(mWWI, mW.' * mY(:));
mX = reshape(vX, size(mR, 2), size(mA, 1));
toc()


0.5 * (norm(mR * mX * mA - mY, 'fro') ^ 2)