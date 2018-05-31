
%% Linear Convolution

numRows = 41;
numCols = 41;

numRowsH = 7;
numColsH = 3;

mH = randn([numRowsH, numColsH]);

mA = randn([numRows, numCols]);

numRowsL = numRows + numRowsH - 1;
numColsL = numCols + numColsH - 1;


mC1 = conv2(mA, mH, 'same');
mC2 = ifft2(fft2(mA, numRowsL, numColsL) .* fft2(mH, numRowsL, numColsL), 'symmetric');

radiusRows = ceil((numRowsH + 1) / 2);
radiusCols = ceil((numColsH + 1) / 2);

mC2 = mC2(radiusRows:(radiusRows + numRows - 1), radiusCols:(radiusCols + numCols - 1));

convErr = norm(mC1(:) - mC2(:), 'inf');
disp(['Convolution Error (Infinity Norm) - ', num2str(convErr)]);


%% Cyclic Convolution

% Image Size
numRows = 41;
numCols = 41;

% Kernel Size
numRowsH = 9;
numColsH = 5;

mH = randn([numRowsH, numColsH]);

if(mod(numRowsH, 2) == 0)
    % Even Number
    mH = [mH; zeros([1, numColsH])];
    numRowsH = numRowsH + 1;
end

if(mod(numColsH, 2) == 0)
    % Even Number
    mH = [mH, zeros([numRowsH, 1])];
    numColsH = numColsH + 1;
end

mA = randn([numRows, numCols]);

radiusRows = floor(numRowsH / 2);
radiusCols = floor(numColsH / 2);

% Cyclic Extension
mHC = mH;
mHC(numRows, numCols) = 0;
mHC = circshift(mHC, [-radiusRows, -radiusCols]);

mAPad = padarray(mA, [radiusRows, radiusCols], 'circular', 'both');

mC1 = conv2(mAPad, mH, 'valid');
mC2 = ifft2(fft2(mA) .* fft2(mHC), 'symmetric');

convErr = norm(mC1(:) - mC2(:), 'inf');
disp(['Convolution Error (Infinity Norm) - ', num2str(convErr)]);


%% Linear Convolution

% Image Size
numRows = 40;
numCols = 40;

% Kernel Size
numRowsH = 8;
numColsH = 6;

% Kernel
mH = randn([numRowsH, numColsH]);
% Image
mA = randn([numRows, numCols]);

% Size of the Linear Convolution Support
numRowsL = numRows + numRowsH - 1;
numColsL = numCols + numColsH - 1;

mC1 = conv2(mA, mH, 'full');
mC2 = ifft2(fft2(mA, numRowsL, numColsL) .* fft2(mH, numRowsL, numColsL), 'symmetric');

convErr = norm(mC1(:) - mC2(:), 'inf');
disp(['Convolution Error (Infinity Norm) - ', num2str(convErr)]);

