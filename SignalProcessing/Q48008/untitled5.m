
% Q19024
clear();
close('all');

wavType = 'haar';

intMethod = 'bicubic';

mImgRef = imread('https://i.imgur.com/8jvEQJX.png'); % Reference image
mImgRef = mImgRef(:, :, 1); %<! Working on a single channel
mImgRef = im2double(mImgRef);
[numRows, numCols] = size(mImgRef);

% Low resolution image
mImg = imresize(mImgRef, 0.5, intMethod, 'Antialiasing', true);

[mLL, mLH, mHL ,mHH] = dwt2(mImg, wavType); % Forward DWT

% Interpolation by 2 factor
mLL1 = imresize(mLL, 2, intMethod);   
mLH1 = imresize(mLH, 2, intMethod);
mHL1 = imresize(mHL, 2, intMethod);
mHH1 = imresize(mHH, 2, intMethod);

mDiffImg = mImg - mLL1;

mLH2 = mLH1 + mDiffImg;
mHL2 = mHL1 + mDiffImg;
mHH2 = mHH1 + mDiffImg;

% alpha = 2 -> No need for resize
mLL3 = imresize(mImg, 1, intMethod);
mLH3 = imresize(mLH2, 1, intMethod);
mHL3 = imresize(mHL2, 1, intMethod);
mHH3 = imresize(mHH2, 1, intMethod);

% High resolution image
mImgHr = idwt2(mLL3, mLH3, mHL3, mHH3, wavType);

% Display high resolution image
figure()
imshow([mImgRef, mImgHr]);

