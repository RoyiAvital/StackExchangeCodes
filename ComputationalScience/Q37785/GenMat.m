clear();

numRows = 21;
numCols = 22;

numRowsK = 5;
numColsK = 7;

mI = rand(numRows, numCols);
mK = rand(numRowsK, numColsK);

mP = padarray(mI, [floor(numRowsK / 2), floor(numColsK / 2)], 'circular', 'both');
mY = conv2(mP, mK, 'valid');

mH = CreateImageFilterMtx(mK, numRows, numCols, 1, 4);

norm(mY(:) - mH * mI(:), inf)

save('Data', 'mI', 'mK', 'mP', 'mY');